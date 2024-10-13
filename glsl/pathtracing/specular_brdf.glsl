
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// "Specular" dielectric microfacet BRDF
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void specular_ndf_roughnesses(out float alpha_x, out float alpha_y)
{
    float rsqr = sqr(specular_roughness);
    alpha_x = rsqr * sqrt(2.0/(1.0 + sqr(1.0 - specular_anisotropy)));
    alpha_y = (1.0 - specular_anisotropy) * alpha_x;
    // (Here opt to clamp to a mininum roughness, rather than deal with a special degenerate case for zero roughness)
    const float min_alpha = 1.0e-4;
    alpha_x = max(min_alpha, alpha_x);
    alpha_y = max(min_alpha, alpha_y);
}

float eta_s()
{
    float coat_ior_average = mix(ambient_ior, coat_ior, coat_weight);
    float eta_s = specular_ior / coat_ior_average;
    return eta_s;
}

float fresnel_refl_normal_incidence()
{
    // Compute Fresnel reflection factor at specular boundary, given presence of coat
    // (before modulation by specular_weight)
    float eta_s = eta_s();
    float F_s = sqr((eta_s - 1.0)/(eta_s + 1.0));
    return F_s;
}

float specular_ior_ratio()
{
    float F_s = fresnel_refl_normal_incidence();
    float xi_s = clamp(specular_weight, 0.0, 1.0/max(F_s, DENOM_TOLERANCE));
    float eta_s = eta_s();
    float tmp = min(1.0, sign(eta_s - 1.0) * sqrt(xi_s * F_s));
    float eta_s_prime = (1.0 + tmp) / max(1.0 - tmp, DENOM_TOLERANCE);
    return eta_s_prime;
}

vec3 fresnel_refl_accounting_for_coat(float mui, float eta_bc)
{
    // Compute correct Fresnel reflection factor at specular boundary, taking into account coat refraction,
    // to avoid TIR artifact:
    //  - mu_i is angle cosine of ray incident from exterior of surface
    //  - eta_bc is the ratio of base specular IOR to coat IOR
    float eta_ca = coat_ior / ambient_ior;
    float eta_ba = specular_ior / ambient_ior;
    // muc is the resulting angle cosine of the ray refracted into the coat, assuming smooth boundaries
    float muc = sqrt(1.0 - (1.0 - sqr(mui))/sqr(eta_ca));
    float eta_ti = mix(eta_ba, eta_bc, coat_weight);
    vec3 F;
    vec3 F_nofilm = vec3(FresnelDielectricReflectance(muc, eta_ti));
#ifdef THIN_FILM_ENABLED
    vec3 F_film = FresnelThinFilmOverDielectric(muc, eta_ti);
    return mix(F_nofilm, F_film, thin_film_weight);
#else
    return F_nofilm;
#endif // THIN_FILM_ENABLED;
}

vec3 specular_brdf_evaluate(in vec3 pW, in Basis basis, in vec3 winputL, in vec3 woutputL,
                            inout float pdf_woutputL)
{
    bool transmitted = woutputL.z * winputL.z < 0.0;
    if (transmitted)
        return vec3(0.0);

    // We assume that the local frame is setup so that the z direction points from the dielectric interior to the exterior.
    // Thus we can determine if the reflection is internal or external to the dielectric:
    vec3 beamOutgoingL = winputL;
    bool external_reflection = (beamOutgoingL.z > 0.0);

    // Compute IOR ratio at interface:
    //  eta_ti_refl = (IOR in hemi. opposite to reflection) / (IOR in hemi. of reflection)
    float eta_ie = specular_ior_ratio(); // n_interior / n_exterior
    float eta_ti_refl = external_reflection ? eta_ie : 1.0/eta_ie;
    if (abs(eta_ti_refl - 1.0) < IOR_EPSILON) // degenerate case of index-matched interface, BRDF goes to zero
        return vec3(0.0);

    // Non-physical Fresnel tint to apply
    vec3 tint = specular_color;

   // Compute the NDF roughnesses in the rotated frame
    float alpha_x, alpha_y;
    specular_ndf_roughnesses(alpha_x, alpha_y);

    // Construct basis such that x, y are aligned with the T, B in the local, rotated frame
    LocalFrameRotation rotation = getLocalFrameRotation(PI2 * specular_rotation);
    vec3 winputR  = localToRotated(winputL,  rotation);
    vec3 woutputR = localToRotated(woutputL, rotation);

    // Compute the micronormal mR in the local (rotated) frame, from the reflection half-vector
    vec3 mR = normalize(woutputR + winputR);

    // Discard backfacing microfacets
    if (dot(mR, winputR) * winputR.z < 0.0 || dot(mR, woutputR) * woutputR.z < 0.0)
        return vec3(0.0);

    // Compute NDF
    float D = ggx_ndf_eval(mR, alpha_x, alpha_y);
    float DV = D * ggx_G1(winputR, alpha_x, alpha_y) * max(0.0, dot(winputR, mR)) / max(DENOM_TOLERANCE, winputR.z);

    // Thus compute PDF of woutputL sample
    float dwh_dwo = 1.0 / max(abs(4.0*dot(winputR, mR)), DENOM_TOLERANCE); // Jacobian of the half-direction mapping
    pdf_woutputL = DV * dwh_dwo;

    // Compute shadowing-masking term
    float G2 = ggx_G2(winputR, woutputR, alpha_x, alpha_y);

    // Compute Fresnel factor for the dielectric reflection
    vec3 F;
#ifdef COAT_ENABLED
    if (external_reflection)
        F = fresnel_refl_accounting_for_coat(abs(dot(winputR, mR)), eta_ti_refl);
    else
#endif // COAT_ENABLED
        F = vec3(FresnelDielectricReflectance(abs(dot(winputR, mR)), eta_ti_refl));

    // Thus evaluate BRDF.
    vec3 f = F * D * G2 / max(4.0 * abs(woutputL.z) * abs(winputL.z), DENOM_TOLERANCE);
    return f * tint;
}


vec3 specular_brdf_sample(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed,
                          out vec3 woutputL, out float pdf_woutputL)
{
    // We assume that the local frame is setup so that the z direction points from the dielectric interior to the exterior.
    // Thus we can determine if the reflection is internal or external to the dielectric:
    vec3 beamOutgoingL = winputL;
    bool external_reflection = (beamOutgoingL.z > 0.0);

    // Compute IOR ratio at interface:
    //  eta_ti_refl = (IOR in hemi. opposite to reflection) / (IOR in hemi. of reflection)
    float eta_ie = specular_ior_ratio(); // n_interior / n_exterior
    float eta_ti_refl = external_reflection ? eta_ie : 1.0/eta_ie;
    if (abs(eta_ti_refl - 1.0) < IOR_EPSILON) // degenerate case of index-matched interface, BRDF goes to zero
        return vec3(0.0);

    // Non-physical Fresnel tint to apply
    vec3 tint = specular_color;

    // Compute the NDF roughnesses in the rotated frame
    float alpha_x, alpha_y;
    specular_ndf_roughnesses(alpha_x, alpha_y);

    // Construct basis such that x, y are aligned with the T, B in the rotated frame
    LocalFrameRotation rotation = getLocalFrameRotation(PI2 * specular_rotation);
    vec3 winputR = localToRotated(winputL, rotation);

    // Sample local microfacet normal mR, according to Heitz "Sampling the GGX Distribution of Visible Normals"
    vec3 mR;
    if (winputR.z > 0.0)
        mR = ggx_ndf_sample(winputR, alpha_x, alpha_y, rndSeed);
    else
    {
        // GGX sampling in negative hemisphere
        vec3 winputR_reflected = winputR;
        winputR_reflected.z *= -1.0;
        mR = ggx_ndf_sample(winputR_reflected, alpha_x, alpha_y, rndSeed);
        mR.z *= -1.0;
    }

    // Compute woutputR (and thus woutputL) by reflecting winputR about mR
    vec3 woutputR = -winputR + 2.0*dot(winputR, mR)*mR;
    if (winputR.z * woutputR.z < 0.0)
    {
        pdf_woutputL = 1.0;
        return vec3(0.0);
    }

    // Rotate woutputR back to local space
    woutputL = rotatedToLocal(woutputR, rotation);

    // Compute NDF, and "distribution of visible normals" DV
    float D = ggx_ndf_eval(mR, alpha_x, alpha_y);
    float DV = D * ggx_G1(winputR, alpha_x, alpha_y) * abs(dot(winputR, mR)) / max(DENOM_TOLERANCE, abs(winputR.z));

    // Thus compute PDF of woutputL sample
    float dwh_dwo = 1.0 / max(abs(4.0*dot(winputR, mR)), DENOM_TOLERANCE); // Jacobian of the half-direction mapping
    pdf_woutputL = DV * dwh_dwo;

    // Compute shadowing-masking term
    float G2 = ggx_G2(winputR, woutputR, alpha_x, alpha_y);

    // Compute Fresnel factor for the dielectric reflection
    vec3 F;
#ifdef COAT_ENABLED
    if (external_reflection)
        F = fresnel_refl_accounting_for_coat(abs(dot(winputR, mR)), eta_ti_refl);
    else
#endif // COAT_ENABLED
        F = vec3(FresnelDielectricReflectance(abs(dot(winputR, mR)), eta_ti_refl));

     // Thus evaluate BRDF.
    vec3 f = F * D * G2 / max(4.0 * abs(woutputL.z) * abs(winputL.z), DENOM_TOLERANCE);
    return f * tint;
}


vec3 specular_brdf_albedo(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed)
{
    // Estimate of the BRDF albedo, used to compute the discrete probability of selecting this lobe
    float eta_ie = specular_ior_ratio(); // n_interior / n_exterior
    if (abs(eta_ie - 1.0) < IOR_EPSILON)
    {
        // degenerate case of index-matched interface, BRDF goes to zero
        return vec3(0.0);
    }

    // Approximate albedo via Monte-Carlo sampling:
    const int num_samples = 1;
    vec3 albedo = vec3(0.0);
    for (int n=0; n<num_samples; ++n)
    {
        vec3 woutputL;
        float pdf_woutputL;
        vec3 f = specular_brdf_sample(pW, basis, winputL, rndSeed, woutputL, pdf_woutputL);
        if (length(f) > RADIANCE_EPSILON)
            albedo += f * abs(woutputL.z) / max(DENOM_TOLERANCE, pdf_woutputL);
    }
    albedo /= float(num_samples);
    return albedo;
}
