precision highp isampler2D;
precision highp usampler2D;
precision highp int;

//////////////////////////////////////////////////////
// varying
//////////////////////////////////////////////////////

varying vec3 vViewPosition;
varying vec3 vWorldPosition;

//////////////////////////////////////////////////////
// camera uniforms
//////////////////////////////////////////////////////

uniform mat4 cameraWorldMatrix;
uniform mat4 invProjectionMatrix;
uniform mat4 invModelMatrix;
uniform vec2 resolution;


//////////////////////////////////////////////////////
// material uniforms
//////////////////////////////////////////////////////

uniform float base_weight;
uniform vec3  base_color;
uniform float base_roughness;
uniform float base_metalness;

uniform float specular_weight;
uniform vec3  specular_color;
uniform float specular_roughness;
uniform float specular_anisotropy;
uniform float specular_rotation;
uniform float specular_ior;

uniform float transmission_weight;
uniform vec3  transmission_color;
uniform float transmission_depth;
uniform vec3  transmission_scatter;
uniform float transmission_scatter_anisotropy;
uniform float transmission_dispersion_abbe_number;
uniform float transmission_dispersion_scale;

uniform float subsurface_weight;
uniform vec3  subsurface_color;
uniform float subsurface_radius;
uniform vec3  subsurface_radius_scale;
uniform float subsurface_anisotropy;

uniform float coat_weight;
uniform vec3  coat_color;
uniform float coat_roughness;
uniform float coat_anisotropy;
uniform float coat_rotation;
uniform float coat_ior;
uniform float coat_darkening;

uniform float fuzz_weight;
uniform vec3  fuzz_color;
uniform float fuzz_roughness;

uniform float thin_film_weight;
uniform float thin_film_thickness;
uniform float thin_film_ior;

uniform float emission_luminance;
uniform vec3  emission_color;

uniform float geometry_opacity;
uniform bool geometry_thin_walled;

//////////////////////////////////////////////////////
// lighting uniforms
//////////////////////////////////////////////////////

uniform float skyPower;
uniform vec3 skyColor;

uniform vec3 sunDir; // in world space
uniform float sunPower;
uniform float sunAngularSize;
uniform vec3 sunColor;
varying vec3 vSunDir;


//////////////////////////////////////////////////////
// useful constants
//////////////////////////////////////////////////////

const float PI                    = 3.141592653589793;
const float RECIPROCAL_PI         = 0.3183098861837907;

// tolerances
const float DENOM_TOLERANCE       = 1.0e-10;
const float PDF_EPSILON           = 1.0e-6;
const float IOR_EPSILON           = 1.0e-5;
const float FLT_EPSILON           = 1.1920929e-7;
const float ALPHA_TOLERANCE       = 1.0e-4;

struct OpenPBRMaterial
{
    float base_weight;
    vec3  base_color;
    float base_roughness;
    float base_metalness;
    float specular_weight;
    vec3  specular_color;
    float specular_roughness;
    float specular_anisotropy;
    float specular_rotation;
    float specular_ior;
    float transmission_weight;
    vec3  transmission_color;
    float transmission_depth;
    vec3  transmission_scatter;
    float transmission_scatter_anisotropy;
    float transmission_dispersion_abbe_number;
    float transmission_dispersion_scale;
    float subsurface_weight;
    vec3  subsurface_color;
    float subsurface_radius;
    vec3  subsurface_radius_scale;
    float subsurface_anisotropy;
    float coat_weight;
    vec3  coat_color;
    float coat_roughness;
    float coat_anisotropy;
    float coat_rotation;
    float coat_ior;
    float coat_darkening;
    float fuzz_weight;
    vec3  fuzz_color;
    float fuzz_roughness;
    float thin_film_weight;
    float thin_film_thickness;
    float thin_film_ior;
    float emission_luminance;
    vec3  emission_color;
    float geometry_opacity;
    bool geometry_thin_walled;
};

void openpbr_init_material(inout OpenPBRMaterial M)
{
    M.base_weight                         = base_weight;
    M.base_color                          = base_color;
    M.base_roughness                      = base_roughness;
    M.base_metalness                      = base_metalness;
    M.specular_weight                     = specular_weight;
    M.specular_color                      = specular_color;
    M.specular_roughness                  = specular_roughness;
    M.specular_anisotropy                 = specular_anisotropy;
    M.specular_rotation                   = specular_rotation;
    M.specular_ior                        = specular_ior;
    M.transmission_weight                 = transmission_weight;
    M.transmission_color                  = transmission_color;
    M.transmission_depth                  = transmission_depth;
    M.transmission_scatter                = transmission_scatter;
    M.transmission_scatter_anisotropy     = transmission_scatter_anisotropy;
    M.transmission_dispersion_abbe_number = transmission_dispersion_abbe_number;
    M.transmission_dispersion_scale       = transmission_dispersion_scale;
    M.subsurface_weight                   = subsurface_weight;
    M.subsurface_color                    = subsurface_color;
    M.subsurface_radius                   = subsurface_radius;
    M.subsurface_radius_scale             = subsurface_radius_scale;
    M.subsurface_anisotropy               = subsurface_anisotropy;
    M.coat_weight                         = coat_weight;
    M.coat_color                          = coat_color;
    M.coat_roughness                      = coat_roughness;
    M.coat_anisotropy                     = coat_anisotropy;
    M.coat_rotation                       = coat_rotation;
    M.coat_ior                            = coat_ior;
    M.coat_darkening                      = coat_darkening;
    M.fuzz_weight                         = fuzz_weight;
    M.fuzz_color                          = fuzz_color;
    M.fuzz_roughness                      = fuzz_roughness;
    M.thin_film_weight                    = thin_film_weight;
    M.thin_film_thickness                 = thin_film_thickness;
    M.thin_film_ior                       = thin_film_ior;
    M.emission_luminance                  = emission_luminance;
    M.emission_color                      = emission_color;
    M.geometry_opacity                    = geometry_opacity;
    M.geometry_thin_walled                = geometry_thin_walled;
}

/////////////////////////////////////////////////////////////////////////
// Transforms and math utils
/////////////////////////////////////////////////////////////////////////

struct Basis
{
    // right-handed coordinate system
    vec3 nW; // aligned with the z-axis in local space
    vec3 tW; // aligned with the x-axis in local space
    vec3 bW; // aligned with the y-axis in local space
};

#define sqr(x) ((x)*(x))

vec3 safe_normalize(in vec3 N)
{
    float l = length(N);
    return N/max(l, DENOM_TOLERANCE);
}

vec3 normalToTangent(in vec3 N)
{
    vec3 T;
    if (abs(N.z) < abs(N.x))
        T = vec3(N.z, 0.0, -N.x);
    else
        T = vec3(0.0, N.z, -N.y);
    T = safe_normalize(T);
    return T;
}

Basis makeBasis(in vec3 nW)
{
    Basis basis;
    basis.nW = safe_normalize(nW);
    basis.tW = normalToTangent(nW);
    basis.bW = cross(basis.nW, basis.tW);
    return basis;
}

vec3 worldToLocal(in vec3 vWorld, in Basis basis)
{
    return vec3( dot(vWorld, basis.tW),
                 dot(vWorld, basis.bW),
                 dot(vWorld, basis.nW) );
}

// V is assumed to be in local (+Z) space.
mat3 orthonormal_basis_ltc(vec3 V)
{
    float lenSqr = dot(V.xy, V.xy);
    vec3 X = lenSqr > 0.0 ? vec3(V.x, V.y, 0.0) * inversesqrt(lenSqr) : vec3(1, 0, 0);
    vec3 Y = vec3(-X.y, X.x, 0.0); // cross(N, X)
    return mat3(X, Y, vec3(0, 0, 1));
}

vec2 fibonacci_lattice(int n, int num_samples)
{
    // provides a stratified, evenly distributed quasi-random set of num_samples points in the 2d unit square
    const float phi = 1.6180339887498948;
    return vec2((float(n) + 0.5) / float(num_samples), fract((float(n) + 1.0) * phi));
}

/////////////////////////////////////////////////////////////////////////
// Microfacet & Fresnel math
/////////////////////////////////////////////////////////////////////////

vec3 FresnelSchlick(vec3 F0, float mu)
{
    return F0 + pow(1.0 - mu, 5.0)*(vec3(1.0) - F0);
}

vec3 FresnelConductorF82(float mu, vec3 F0, vec3 F82)
{
    const float mu_bar = 1.0/7.0;
    const float denom = mu_bar * pow(1.0 - mu_bar, 6.0);
    vec3 Fschlick_bar = FresnelSchlick(F0, mu_bar);
    vec3 Fschlick     = FresnelSchlick(F0, mu);
    return Fschlick - mu * pow(1.0 - mu, 6.0) * (vec3(1.0) - F82) * Fschlick_bar / denom;
}

// Dielectric Fresnel factor.
// mui     = magnitude of the cosine of the incident ray angle to the micronormal
// eta_ti  = ratio et/ei of the transmitted IOR (et) and incident IOR (ei)
// Outputs vec2(rs, rp), the amplitudes of the S, P polarized reflection (where squared amplitudes give reflectance).
vec2 FresnelDielectricPolarizations(float mui, float eta_ti)
{
    float mut2 = sqr(eta_ti) - (1.0 - sqr(mui));
    if (mut2 <= 0.0) return vec2(1.0, 1.0); // (total internal reflection)
    float mut = sqrt(mut2) / eta_ti;
    float rs = (mui - eta_ti*mut) / (mui + eta_ti*mut);
    float rp = (mut - eta_ti*mui) / (mut + eta_ti*mui);
    return vec2(rs, rp);
}

// mui     = magnitude of the cosine of the incident ray angle to the micronormal
// eta_ti  = ratio et/ei of the transmitted IOR (et) and incident IOR (ei)
float FresnelDielectricReflectance(in float mui, in float eta_ti)
{
    // assuming unpolarized incident light
    vec2 r = FresnelDielectricPolarizations(mui, eta_ti);
    return 0.5*dot(r, r);
}

// Dielectric Fresnel factor, at normal incidence.
float FresnelDielectricReflectanceNormal(float eta_ti)
{
    float etamo = eta_ti - 1.0;
    float etapo = eta_ti + 1.0;
    return sqr(etamo)/sqr(etapo);
}

float FresnelDielectricReflectanceAverage(float eta)
{
    // Hemispherical albedo of dielectric Fresnel
    if (eta < 1.03)
        return (eta - 1.0)/3.0; // limiting formula at low eta, accurate to < 0.2%
    // exact formula (from d'Eon's Hitchhiker's Guide)
    float etapo = eta + 1.0; float etamo = eta - 1.0;
    float f_perp = 8.0/3.0*(etapo-0.5)/sqr(etapo);
    float eta2 = sqr(eta); float eta3 = eta*eta2;
    float eta2po = eta2 + 1.0; float eta2po2 = sqr(eta2po); float eta2po3 = eta2po2*eta2po;
    float eta2mo = eta2 - 1.0; float eta2mo2 = sqr(eta2mo);
    float f_para = 2.0*eta2*(2.0*eta*(eta2+2.0*eta-1.0)/(eta2po2*eta2mo) - eta2po*log(eta)/eta2mo2 + eta2mo2*log(eta*etapo/etamo)/eta2po3);
    return 1.0 - 0.5*(f_perp + f_para);
}

// Given the direction (wt) of a light beam transmitted through a plane dielectric interface
// with the given normal (n) in any orientation, and the ratio eta_ti_photon = nt/ni of the transmitted IOR (nt) and incident IOR (ni),
// compute the direction of the incident light beam (wi). Returns false if no such beam exists, due to total internal reflection.
bool FresnelDielectricRefraction(in vec3 n, in float eta_ti_photon, in vec3 wt, inout vec3 wi)
{
    float wtn = dot(wt, n);
    float disciminant = 1.0 - sqr(eta_ti_photon)*(1.0 - sqr(wtn));
    if (disciminant < 0.0) return false;
    wi = eta_ti_photon*wt - n*sign(wtn)*(eta_ti_photon*abs(wtn) - sqrt(disciminant));
    return true;
}

float ggx_ndf_eval(in vec3 m, in vec2 alpha)
{
    // Evaluate the anisotropic GGX NDF
    float Ddenom = PI * alpha.x * alpha.y * sqr(sqr(m.x/alpha.x) + sqr(m.y/alpha.y) + sqr(m.z));
    return 1.0 / max(Ddenom, DENOM_TOLERANCE);
}

// GGX NDF sampling routine, as described in
//  "Sampling Visible GGX Normals with Spherical Caps", Dupuy et al., HPG 2023.
// NB, this assumes V is in the +z hemisphere, and returns a sampled micronormal in that hemisphere.
vec3 ggx_ndf_sample(in vec3 V, in vec2 alpha, in vec2 xi)
{
    // Transform the view direction to the hemisphere configuration.
    V = normalize(vec3(V.xy * alpha, V.z));
    // Sample a spherical cap in (-V.z, 1].
    float phi = 2.0 * PI * xi.x;
    float z = (1.0 - xi.y) * (1.0 + V.z) - V.z;
    float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    vec3 c = vec3(x, y, z);
    // Compute the microfacet normal.
    vec3 H = c + V;
    // Transform the microfacet normal back to the ellipsoid configuration.
    H = normalize(vec3(H.xy * alpha, H.z));
    return H;
}

float ggx_lambda(in vec3 V, in vec2 alpha)
{
    return (-1.0 + sqrt(1.0 + (sqr(alpha.x*V.x) + sqr(alpha.y*V.y))/sqr(V.z))) / 2.0;
}

float ggx_G1(in vec3 V, in vec2 alpha)
{
    if (abs(V.z) < FLT_EPSILON) return 0.0;
    return 1.0 / (1.0 + ggx_lambda(V, alpha));
}

// Height-correlated form of GGX shadowing-masking function
float ggx_G2(in vec3 V, in vec3 L, in vec2 alpha)
{
    if (abs(V.z) < FLT_EPSILON || abs(L.z) < FLT_EPSILON) return 0.0;
    return 1.0 / (1.0 + ggx_lambda(V, alpha) + ggx_lambda(L, alpha));
}


///////////////////////////////////////////////////////////////////////
// Fuzz BRDF
///////////////////////////////////////////////////////////////////////

// Gaussian fit to directional albedo table.
float zeltner_sheen_dir_albedo(float x, float y)
{
    float s = y*(0.0206607 + 1.58491*y)/(0.0379424 + y*(1.32227 + y));
    float m = y*(-0.193854 + y*(-1.14885 + y*(1.7932 - 0.95943*y*y)))/(0.046391 + y);
    float o = y*(0.000654023 + (-0.0207818 + 0.119681*y)*y)/(1.26264 + y*(-1.92021 + y));
    return exp(-0.5*sqr((x - m)/s))/(s*sqrt(2.0*PI)) + o;
}

// Rational fits to LTC matrix coefficients.
float zeltner_sheen_ltc_aInv(float x, float y)
{
    return (2.58126*x + 0.813703*y)*y/(1.0 + 0.310327*x*x + 2.60994*x*y);
}
float zeltner_sheen_ltc_bInv(float x, float y)
{
    return sqrt(1.0 - x)*(y - 1.0)*y*y*y/(0.0000254053 + 1.71228*x - 1.71506*x*y + 1.34174*y*y);
}

vec3 fuzz_brdf_evaluate(in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    float NdotV = min(V.z, 1.0);
    float r = clamp(pbr.fuzz_roughness, 0.01, 1.0); // Clamp to the range of the original impl.
    mat3 toLTC = transpose(orthonormal_basis_ltc(V));
    vec3 w = toLTC * L;

    // Transform w to original configuration (clamped cosine).
    //                 |aInv    0 bInv|
    // wo = M^-1 . w = |   0 aInv    0| . w
    //                 |   0    0    1|
    float aInv = zeltner_sheen_ltc_aInv(NdotV, r);
    float bInv = zeltner_sheen_ltc_bInv(NdotV, r);
    vec3 wo = vec3(aInv*w.x + bInv*w.z, aInv * w.y, w.z);
    float lenSqr = dot(wo, wo);

    // D(w) = Do(M^-1.w / ||M^-1.w||) . |M^-1| / ||M^-1.w||^3
    //      = Do(M^-1.w) . |M^-1| / ||M^-1.w||^4
    //      = Do(wo) . |M^-1| / dot(wo, wo)^2
    //      = Do(wo) . aInv^2 / dot(wo, wo)^2
    //      = Do(wo) . (aInv / dot(wo, wo))^2
    float jacobian = sqr(aInv / lenSqr);
    float pdfL = max(wo.z, 0.0) * RECIPROCAL_PI * jacobian;
    float albedo = zeltner_sheen_dir_albedo(NdotV, r);
    float NdotL = max(abs(L.z), FLT_EPSILON);
    return pbr.fuzz_color * albedo * pdfL / NdotL;
}

vec3 fuzz_brdf_albedo(in vec3 V, const in OpenPBRMaterial pbr)
{
    return vec3(zeltner_sheen_dir_albedo(V.z, pbr.fuzz_roughness));
}


///////////////////////////////////////////////////////////////////////
// coat/specular dielectric BRDFs
///////////////////////////////////////////////////////////////////////

struct DielectricBSDFParams
{
    float eta_ie; // internal/external IOR ratio
    vec2 alpha;   // alpha roughnesses
};

vec3 dielectric_brdf_evaluate(in vec3 V, in vec3 L, const in DielectricBSDFParams P)
{
    vec3 H = normalize(V + L); // Micronormal
    if (dot(H, V) < 0.0 || dot(H, L) < 0.0)
        return vec3(0.0); // Discard backfacing microfacets
    float F = FresnelDielectricReflectance(abs(dot(V, H)), P.eta_ie); // Dielectric Fresnel
    float D = ggx_ndf_eval(H, P.alpha); // NDF
    float G2 = ggx_G2(V, L, P.alpha); // Shadowing-masking term
    float J = 1.0 / max(4.0 * abs(V.z) * abs(L.z), DENOM_TOLERANCE); // Jacobian of half-direction mapping
    float f = F * D * G2 * J;
    return vec3(f);
}

vec3 dielectric_brdf_albedo(in vec3 V, const in DielectricBSDFParams P)
{
    // Approximate albedo via (deterministic) Monte-Carlo sampling:
    const int num_samples = 16;
    float albedo = 0.0;
    for (int n=0; n<num_samples; ++n)
    {
        vec2 xi = fibonacci_lattice(n, num_samples);
        vec3 L = ggx_ndf_sample(V, P.alpha, xi);
        float G2 = ggx_G2(V, L, P.alpha);
        float G1 = ggx_G1(V, P.alpha);
        vec3 H = normalize(V + L);
        float F = FresnelDielectricReflectance(abs(dot(V, H)), P.eta_ie);
        albedo += F * G2 /  max(G1, DENOM_TOLERANCE);
    }
    albedo /= float(num_samples);
    return clamp(vec3(albedo), vec3(0.0), vec3(1.0));
}

vec2 coat_ndf_roughnesses(const in OpenPBRMaterial pbr)
{
    float alpha_x = max(ALPHA_TOLERANCE, sqr(pbr.coat_roughness) * sqrt(2.0/(1.0 + sqr(1.0 - pbr.coat_anisotropy))));
    float alpha_y = max(ALPHA_TOLERANCE, (1.0 - pbr.coat_anisotropy) * alpha_x);
    return vec2(alpha_x, alpha_y);
}

vec3 coat_brdf_evaluate(in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    DielectricBSDFParams P;
    P.eta_ie = pbr.coat_ior; // (assuming here that ambient IOR is 1)
    P.alpha  = coat_ndf_roughnesses(pbr);
    return dielectric_brdf_evaluate(V, L, P);
}

vec3 coat_brdf_albedo(in vec3 V, const in OpenPBRMaterial pbr)
{
    DielectricBSDFParams P;
    P.eta_ie = pbr.coat_ior; // (assume ambient IOR is 1)
    P.alpha  = coat_ndf_roughnesses(pbr);
    return dielectric_brdf_albedo(V, P);
}

float specular_brdf_ior_ratio(const in OpenPBRMaterial pbr)
{
    // Compute IOR ratio at specular boundary for BRDF calculation, accounting for coat
    float eta_sc = pbr.specular_ior / pbr.coat_ior;
    if (eta_sc < 1.0) eta_sc = 1.0/eta_sc; // (flip spec/coat IOR ratio if needed to keep it > 1, to correct for refraction in coat)
    float eta_s = mix(pbr.specular_ior, eta_sc, coat_weight); // (assuming here that ambient IOR is 1)
    float F_s = sqr((eta_s - 1.0)/(eta_s + 1.0)); // Fresnel at normal incidence
    float tmp = min(1.0, sign(eta_s - 1.0) * sqrt(clamp(pbr.specular_weight * F_s, 0.0, 1.0)));
    return (1.0 + tmp) / max(1.0 - tmp, DENOM_TOLERANCE); // modulated IOR ratio
}

float specular_ior_modulated(const in OpenPBRMaterial pbr)
{
    // Compute physical IOR of base dielectric, according to specular_weight modulation
    float eta_s = pbr.specular_ior;
    float F_s = sqr((eta_s - 1.0)/(eta_s + 1.0)); // Fresnel at normal incidence
    float tmp = min(1.0, sign(eta_s - 1.0) * sqrt(clamp(pbr.specular_weight * F_s, 0.0, 1.0)));
    return (1.0 + tmp) / max(1.0 - tmp, DENOM_TOLERANCE); // modulated IOR ratio
}

vec2 specular_ndf_roughnesses(const in OpenPBRMaterial pbr)
{
    float alpha_x = max(ALPHA_TOLERANCE, sqr(pbr.specular_roughness) * sqrt(2.0/(1.0 + sqr(1.0 - pbr.specular_anisotropy))));
    float alpha_y = max(ALPHA_TOLERANCE, (1.0 - pbr.specular_anisotropy) * alpha_x);
    return vec2(alpha_x, alpha_y);
}

vec3 specular_brdf_evaluate(in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    DielectricBSDFParams P;
    P.eta_ie = specular_brdf_ior_ratio(pbr);
    P.alpha  = specular_ndf_roughnesses(pbr);
    return dielectric_brdf_evaluate(V, L, P);
}

vec3 specular_brdf_albedo(in vec3 V, const in OpenPBRMaterial pbr)
{
    DielectricBSDFParams P;
    P.eta_ie = specular_brdf_ior_ratio(pbr);
    P.alpha  = specular_ndf_roughnesses(pbr);
    return dielectric_brdf_albedo(V, P);
}


///////////////////////////////////////////////////////////////////////
// Metal BRDF
///////////////////////////////////////////////////////////////////////

vec3 metal_brdf_evaluate(in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    vec3 H = normalize(V + L); // Micronormal
    vec3 F0  = pbr.base_weight * pbr.base_color;
    vec3 F82 = pbr.specular_color;
    vec3 F = FresnelConductorF82(abs(dot(V, H)), F0, F82); // Conductor Fresnel
    vec2 alpha = specular_ndf_roughnesses(pbr);
    float D = ggx_ndf_eval(H, alpha); // NDF
    float G2 = ggx_G2(V, L, alpha); // Shadowing-masking term
    float J = 1.0 / max(4.0 * abs(V.z) * abs(L.z), DENOM_TOLERANCE); // Jacobian of half-direction mapping
    vec3 f = min(vec3(1.0), pbr.specular_weight*F) * D * G2 * J;
    return f;
}

vec3 metal_brdf_albedo(in vec3 V, const in OpenPBRMaterial pbr)
{
    // Approximate albedo via (deterministic) Monte-Carlo sampling:
    const int num_samples = 16;
    vec2 alpha = specular_ndf_roughnesses(pbr);
    vec3 albedo = vec3(0.0);
    for (int n=0; n<num_samples; ++n)
    {
        vec2 xi = fibonacci_lattice(n, num_samples);
        vec3 L = ggx_ndf_sample(V, alpha, xi);
        float G2 = ggx_G2(V, L, alpha);
        float G1 = ggx_G1(V, alpha);
        vec3 H = normalize(V + L);
        vec3 F0  = pbr.base_weight * pbr.base_color;
        vec3 F82 = pbr.specular_color;
        vec3 F = FresnelConductorF82(abs(dot(V, H)), F0, F82);
        albedo += min(vec3(1.0), pbr.specular_weight*F) * G2 /  max(G1, DENOM_TOLERANCE);
    }
    albedo /= float(num_samples);
    return clamp(albedo, vec3(0.0), vec3(1.0));
}


///////////////////////////////////////////////////////////////////////
// Specular BTDF
///////////////////////////////////////////////////////////////////////

vec3 specular_btdf_evaluate(in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    bool reflected = V.z * L.z > 0.0;
    if (reflected)
        return vec3(0.0);
    float eta_ie = specular_ior_modulated(pbr); // (assuming here that ambient IOR is 1)
    float eta_ti_photon = 1.0 / eta_ie;
    // Compute the micronormal mR in the local frame such that -L is refracted to V
    vec3 beamIncidentR = -L;
    vec3 beamOutgoingR = V;
    vec3 mR = beamIncidentR - eta_ti_photon*beamOutgoingR;
    if (dot(mR, mR) == 0.0)
        return vec3(0.0);
    vec3 H = safe_normalize(mR);
    if (H.z <= 0.0) H *= -1.0; // orient the micronormal into the positive hemisphere
    vec3 tint = (pbr.transmission_depth == 0.0) ? pbr.transmission_color : vec3(1.0); // non-physical tint
    float T = max(0.0, 1.0 - FresnelDielectricReflectance(abs(dot(V, H)), eta_ie)); // dielectric Fresnel transmittance
    if (abs(eta_ti_photon - 1.0) < IOR_EPSILON)
        return tint * T;
    float im = dot(-beamIncidentR, H);
    float J; // Jacobian of half-direction mapping
    {
        float om = dot(beamOutgoingR, H);
        J = sqr(eta_ti_photon) * abs(om) / max(sqr(im + eta_ti_photon*om), DENOM_TOLERANCE);
    }
    vec2 alpha = specular_ndf_roughnesses(pbr);
    float D = ggx_ndf_eval(H, alpha); // NDF
    float G2 = ggx_G2(V, L, alpha); // Shadowing-masking term
    vec3 f = tint * T * abs(dot(V, H)) * J * G2 * D / max(abs(L.z) * abs(V.z), DENOM_TOLERANCE);
    return f;
}

vec3 specular_btdf_albedo(in vec3 V, const in OpenPBRMaterial pbr)
{
    DielectricBSDFParams P;
    float eta_ie = specular_ior_modulated(pbr); // (assuming here that ambient IOR is 1)
    float eta_ti_photon = 1.0 / eta_ie;
    vec2   alpha = specular_ndf_roughnesses(pbr);
    // Approximate albedo via (deterministic) Monte-Carlo sampling:
    const int num_samples = 16;
    float albedo = 0.0;
    for (int n=0; n<num_samples; ++n)
    {
        vec2 xi = fibonacci_lattice(n, num_samples);
        vec3 L = ggx_ndf_sample(V, alpha, xi);
        float G2 = ggx_G2(V, L, alpha);
        float G1 = ggx_G1(V, alpha);
        vec3 H = -normalize(L + eta_ti_photon*V);
        float T = max(0.0, 1.0 - FresnelDielectricReflectance(abs(dot(V, H)), eta_ie));
        albedo += T * G2 /  max(G1, DENOM_TOLERANCE);
    }
    albedo /= float(num_samples);
    vec3 tint = (pbr.transmission_depth == 0.0) ? pbr.transmission_color : vec3(1.0); // non-physical tint
    return clamp(tint*vec3(albedo), vec3(0.0), vec3(1.0));
}


///////////////////////////////////////////////////////////////////////
// Diffuse BRDF - EON model
///////////////////////////////////////////////////////////////////////

#define constant1_FON (0.5 - 2.0/(3.0*PI))
#define constant2_FON (2.0/3.0 - 28.0/(15.0*PI))

// FON directional albedo (approx.)
float E_FON_approx(float mu, float r)
{
    float mucomp = 1.0f - mu;
    float mucomp2 = mucomp * mucomp;
    const mat2 Gcoeffs = mat2(0.0571085289f, -0.332181442f, 0.491881867f, 0.0714429953f);
    float GoverPi = dot(Gcoeffs * vec2(mucomp, mucomp2), vec2(1.0f, mucomp2));
    return (1.0f + r * GoverPi) / (1.0f + constant1_FON * r);
}

vec3 diffuse_brdf_evaluate(in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    vec3 rho = pbr.base_weight * pbr.base_color;
    float r = pbr.base_roughness;
    float s = dot(V, L) - V.z * L.z;                 // QON s term
    float stinv = s > 0.0 ? s / max(V.z, L.z) : s;   // Fujii model stinv
    float AF = 1.0 / (1.0 + constant1_FON*r);        // Fujii model A coefficient
    vec3 f_ss = (rho / PI) * AF * (1.0 + r*stinv);   // single-scatt. BRDF
    float EFo = E_FON_approx(L.z, r);                // EFo at rho=1 (approx)
    float EFi = E_FON_approx(V.z, r);                // EFi at rho=1 (approx)
    float avgEF = AF * (1.0 + constant2_FON*r);      // avg. albedo
    vec3 rho_ms = sqr(rho) * avgEF / (vec3(1.0) - rho*max(0.0, 1.0-avgEF));
    vec3 f_ms = (rho_ms / PI) * max(FLT_EPSILON, 1.0 - EFo) *   // multi-scatter lobe
                                max(FLT_EPSILON, 1.0 - EFi) /
                                max(FLT_EPSILON, 1.0 - avgEF);
    return f_ss + f_ms;
}

vec3 diffuse_brdf_albedo(in vec3 V, const in OpenPBRMaterial pbr)
{
    vec3 rho = pbr.base_weight * pbr.base_color;
    float r = pbr.base_roughness;
    float muI = V.z;                                 // input angle cos
    float AF = 1.0 / (1.0 + constant1_FON*r);        // FON model A coefficient
    float EF = E_FON_approx(muI, r);                 // EFi at rho=1 (approx)
    float avgEF = AF * (1.0 + constant2_FON*r);      // average albedo
    vec3 rho_ms = sqr(rho) * avgEF / (vec3(1.0) - rho*max(0.0, 1.0-avgEF));
    return rho*EF + rho_ms*(1.0 - EF);
}


///////////////////////////////////////////////////////////////////////
// OpenPBR BSDF
///////////////////////////////////////////////////////////////////////

const int ID_FUZZ_BRDF = 0;
const int ID_COAT_BRDF = 1;
const int ID_META_BRDF = 2;
const int ID_SPEC_BRDF = 3;
const int ID_SPEC_BTDF = 4;
const int ID_DIFF_BRDF = 5;
const int ID_SSSC_BTDF = 6;
const int NUM_LOBES    = 7;

struct OpenPBRLobeWeights
{
    // Weight multipliers of individual BSDF lobes
    vec3 w[7];
};

void openpbr_lobe_weights(in vec3 V, const in OpenPBRMaterial pbr,
                          inout OpenPBRLobeWeights W)
{
    float F  = pbr.fuzz_weight;
    float C  = pbr.coat_weight;
    float M  = pbr.base_metalness;
    float T  = pbr.transmission_weight;
    float S  = pbr.subsurface_weight;
    float xi = pbr.specular_weight;
    bool has_fuzz              = (F > 0.0);
    bool has_coat              = (C > 0.0);
    bool has_metal             = (M > 0.0);
    bool has_dielectric        = (M < 1.0);
    bool has_transmission      = has_dielectric && (T > 0.0);
    bool has_dielectric_opaque = has_dielectric && (T < 1.0);
    bool has_diffuse           = has_dielectric_opaque && (S < 1.0);
    bool has_subsurface        = has_dielectric_opaque && (S > 0.0);

    // compute albedos of present slabs
    vec3  fuzz_albedo = has_fuzz         ? fuzz_brdf_albedo(V, pbr)         : vec3(0.0);
    vec3  coat_albedo = has_coat         ? coat_brdf_albedo(V, pbr)         : vec3(0.0);
    vec3  spec_albedo = has_diffuse      ? specular_brdf_albedo(V, pbr)     : vec3(0.0);
    vec3 trans_albedo = has_transmission ? vec3(1.0) - spec_albedo          : vec3(0.0);
    vec3  meta_albedo = has_metal        ? metal_brdf_albedo(V, pbr)        : vec3(0.0);
    vec3  diff_albedo = has_diffuse      ? pbr.base_weight * pbr.base_color : vec3(0.0);
    vec3   sss_albedo = has_subsurface   ? pbr.subsurface_color             : vec3(0.0);

    // calculate slab weights according to layer structure
    vec3 w_coated_base = mix(vec3(1.0), vec3(1.0) - fuzz_albedo, F);
    vec3 w_base_substrate;
    if (has_coat)
    {
        vec3 base_darkening = vec3(1.0);
        if (coat_darkening > 0.0)
        {
            float Kr = 1.0 - (1.0 - FresnelDielectricReflectanceAverage(pbr.coat_ior))/sqr(pbr.coat_ior);
            float Ks = FresnelDielectricReflectance(abs(V.z), pbr.coat_ior);
            float Fs = FresnelDielectricReflectanceNormal(specular_brdf_ior_ratio(pbr));
            float rd = mix(1.0, pbr.specular_roughness, min(1.f, xi*Fs)); // estimate of roughness of dielectric base
            float rm = pbr.specular_roughness;                            // roughness of metallic base
            float rb = mix(rd, rm, M);                                    // thus estimated roughness of entire base
            float K = mix(Ks, Kr, rb);  // thus estimated internal diffuse reflection coeff.
            vec3 E_dielectric_base = mix(mix(diff_albedo, sss_albedo, S), trans_albedo, T); // dielectric base albedo
            vec3 E_base = mix(E_dielectric_base, meta_albedo, M);                           // entire base albedo
            vec3 Delta = (1.0 - K) / vec3(1.0 - E_base*K);              // full darkening factor
            base_darkening = mix(vec3(1.0), Delta, C * coat_darkening); // modulated darkening factor
        }
        w_base_substrate = w_coated_base * mix(vec3(1.0), base_darkening * pbr.coat_color * (vec3(1.0) - coat_albedo), C);
    }
    else
        w_base_substrate = w_coated_base;

    vec3 w_dielectric_base        = w_base_substrate * vec3(max(0.0, 1.0 - M));
    vec3 w_opaque_dielectric_base = w_dielectric_base * (1.0 - T);
    vec3 w_glossy_diffuse_base    = w_opaque_dielectric_base * (vec3(1.0) - spec_albedo);

    // resulting lobe weights
    W.w[ID_FUZZ_BRDF] = vec3(F);
    W.w[ID_COAT_BRDF] = w_coated_base * C;
    W.w[ID_META_BRDF] = w_base_substrate * M;
    W.w[ID_SPEC_BRDF] = w_dielectric_base * pbr.specular_color;
    W.w[ID_SPEC_BTDF] = w_dielectric_base * T;
    W.w[ID_SSSC_BTDF] = w_opaque_dielectric_base * S;
    W.w[ID_DIFF_BRDF] = w_glossy_diffuse_base * (1.0 - S);
}

vec3 openpbr_lobe_eval(int lobe_id, in vec3 V, in vec3 L, const in OpenPBRMaterial pbr)
{
    switch (lobe_id)
    {
        case ID_FUZZ_BRDF: return     fuzz_brdf_evaluate(V, L, pbr);
        case ID_COAT_BRDF: return     coat_brdf_evaluate(V, L, pbr);
        case ID_META_BRDF: return    metal_brdf_evaluate(V, L, pbr);
        case ID_SPEC_BRDF: return specular_brdf_evaluate(V, L, pbr);
        case ID_DIFF_BRDF: return  diffuse_brdf_evaluate(V, L, pbr);
        case ID_SPEC_BTDF:
        case ID_SSSC_BTDF: return specular_btdf_evaluate(V, L, pbr);
    }
}

vec3 openpbr_bsdf(const in vec3 V, const in vec3 L,
                  const in OpenPBRLobeWeights W, const in OpenPBRMaterial pbr)
{
    if (L.z < DENOM_TOLERANCE || V.z < DENOM_TOLERANCE) return vec3(0.0);
    vec3 f = vec3(0.0);
    for (int lobe_id=0; lobe_id<NUM_LOBES; ++lobe_id)
    {
        vec3 w = W.w[lobe_id];
        if (any(greaterThan(w, vec3(0.f))))
            f += w * openpbr_lobe_eval(lobe_id, V, L, pbr);
    }
    return f;
}

vec3 openpbr_lobe_albedo(int lobe_id, in vec3 V,
                         const in OpenPBRLobeWeights W, const in OpenPBRMaterial pbr)
{
    if (V.z < DENOM_TOLERANCE) return vec3(0.0);
    vec3 w = W.w[lobe_id];
    if (!any(greaterThan(w, vec3(0.f))))
        return vec3(0.0);
    switch (lobe_id)
    {
        case ID_FUZZ_BRDF: return     fuzz_brdf_albedo(V, pbr) * w;
        case ID_COAT_BRDF: return     coat_brdf_albedo(V, pbr) * w;
        case ID_META_BRDF: return    metal_brdf_albedo(V, pbr) * w;
        case ID_SPEC_BRDF: return specular_brdf_albedo(V, pbr) * w;
        case ID_DIFF_BRDF: return  diffuse_brdf_albedo(V, pbr) * w;
        case ID_SPEC_BTDF:
        case ID_SSSC_BTDF: return specular_btdf_albedo(V, pbr) * w;
    }
}


///////////////////////////////////////////////////////////////////////
// Fragment shader main
///////////////////////////////////////////////////////////////////////

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <envmap_common_pars_fragment>
#include <shadowmap_pars_fragment>

void main()
{
    #include <map_fragment>
	#include <color_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>

    // Set up vertex frame
    vec3 Nworld = normalize(inverseTransformDirection(normal, viewMatrix));
    Basis basis = makeBasis(Nworld);

    vec3 Vworld = normalize(cameraPosition - vWorldPosition);
    vec3 Vlocal = worldToLocal(Vworld, basis); // (BRDF evaluation happens in the local space of the vertex frame)

    // Initialize material
    OpenPBRMaterial pbr;
    openpbr_init_material(pbr);
    pbr.specular_roughness = max(0.05, pbr.specular_roughness);

    OpenPBRLobeWeights W;
    openpbr_lobe_weights(Vlocal, pbr, W);

    // Compute direct lighting
    vec3 radiance = vec3(0.0);
    {
        // Sun contribution
        {
            // Query shadow map for sun visibility
            vec3 sun_visibility = vec3(1.0);
            #if ( NUM_DIR_LIGHTS > 0 )
            #pragma unroll_loop_start
                for (int i = 0; i < NUM_DIR_LIGHTS; i++)
                {
                    DirectionalLight directionalLight = directionalLights[i];
                    IncidentLight directLight;
                    getDirectionalLightInfo( directionalLight, directLight );
                    #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
                    DirectionalLightShadow directionalLightShadow = directionalLightShadows[ i ];
                    sun_visibility *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[i],
                                                                                            directionalLightShadow.shadowMapSize,
                                                                                            directionalLightShadow.shadowBias,
                                                                                            directionalLightShadow.shadowRadius,
                                                                                            vDirectionalShadowCoord[i] ) : 1.0;
                    #endif
                }
            #pragma unroll_loop_end
            #endif

            vec3 Lworld = normalize(sunDir);
            float NdotL = max(0.0, dot(Nworld, Lworld));
            sun_visibility *= NdotL > 0.0 ? 1.0 : 0.0;
            if (length(sun_visibility) > 0.0)
            {
                vec3 Llocal = worldToLocal(Lworld, basis);
                vec3 f = openpbr_bsdf(Vlocal, Llocal, W, pbr);
                radiance += sun_visibility * f * NdotL * sunColor * sunPower;
            }
        }

        // Sky contribution
        {
            vec3 Lworld_reflect = normalize(-Vworld + 2.0*dot(Nworld, Vworld)*Nworld); // query env map at direction reflect(V, N)

            // diffuse contribution
            {
                vec4 diff_env = textureLod(envMap, vec3(0.0, 1.0, 0.0), 1000.0);
                vec3 diffuse_albedo = vec3(0.0);
                diffuse_albedo += openpbr_lobe_albedo(ID_DIFF_BRDF, Vlocal, W, pbr);
                diffuse_albedo += openpbr_lobe_albedo(ID_FUZZ_BRDF, Vlocal, W, pbr);
                radiance += skyColor * skyPower * diff_env.rgb * diffuse_albedo;
            }

            ivec2 env_res = textureSize(envMap, 0);
            int env_max_res = max(env_res.x, env_res.y);

            // specular contribution
            {
                float lod = specular_roughness > 0.0 ? max(0.0, 1.0 + log2(float(env_max_res)*sqr(specular_roughness))) : 0.0;
                vec4 spec_env = textureLod(envMap, vec3(Lworld_reflect.x, Lworld_reflect.yz), lod);
                vec3 specular_albedo = vec3(0.0);
                specular_albedo += openpbr_lobe_albedo(ID_SPEC_BRDF, Vlocal, W, pbr);
                specular_albedo += openpbr_lobe_albedo(ID_META_BRDF, Vlocal, W, pbr);
                radiance += skyColor * skyPower * spec_env.rgb * specular_albedo;
            }

            // coat contribution
            {
                float lod = coat_roughness > 0.0 ? max(0.0, 1.0 + log2(float(env_max_res)*sqr(coat_roughness))) : 0.0;
                vec4 spec_env = textureLod(envMap, vec3(Lworld_reflect.x, Lworld_reflect.yz), lod);
                vec3 coat_albedo = openpbr_lobe_albedo(ID_COAT_BRDF, Vlocal, W, pbr);
                radiance += skyColor * skyPower * spec_env.rgb * coat_albedo;
            }

            // transmission contribution
            {
                float eta_ie = specular_ior_modulated(pbr); // (assuming here that ambient IOR is 1)
                float eta_ti_photon = 1.0 / eta_ie;
                vec3 beamOutgoingW = Vworld;
                vec3 beamIncidentW; // the incident photon direction, to be determined (where woutputR = -beamIncidentR)
                if (FresnelDielectricRefraction(Nworld, eta_ti_photon, beamOutgoingW, beamIncidentW))
                {
                    vec3 Lworld_refract = -safe_normalize(beamIncidentW);
                    float lod = specular_roughness > 0.0 ? max(0.0, 1.0 + log2(float(env_max_res)*sqr(specular_roughness))) : 0.0;
                    vec4 spec_env = textureLod(envMap, vec3(Lworld_refract.x, Lworld_refract.yz), lod);
                    vec3 specular_albedo = openpbr_lobe_albedo(ID_SPEC_BTDF, Vlocal, W, pbr);
                    radiance += skyColor * skyPower * spec_env.rgb * specular_albedo;
                }
            }
        }
    }

    gl_FragColor.rgb = radiance;
    gl_FragColor.a = 1.0;

}
