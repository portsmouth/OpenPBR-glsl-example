

/////////////////////////////////////////////////////////////////////////
// Raytracing routines
/////////////////////////////////////////////////////////////////////////

bool bvhIntersectFirstHitWithinDistance(
	BVH bvh, vec3 rayOrigin, vec3 rayDirection, in float maxDistance,
	// output variables
	inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord,
	inout float side, inout float dist)
{
	// stack needs to be twice as long as the deepest tree we expect because
	// we push both the left and right child onto the stack every traversal
	int ptr = 0;
	uint stack[ 60 ];
	stack[ 0 ] = 0u;
	float triangleDistance = 1e20;
	bool found = false;
	while (ptr > - 1 && ptr < 60)
    {
		uint currNodeIndex = stack[ ptr ];
		ptr --;
		// check if we intersect the current bounds
		float boundsHitDistance = intersectsBVHNodeBounds( rayOrigin, rayDirection, bvh, currNodeIndex );
		if (boundsHitDistance == INFINITY ||
            boundsHitDistance > triangleDistance ||
            boundsHitDistance > maxDistance)
		        continue;
		uvec2 boundsInfo = uTexelFetch1D( bvh.bvhContents, currNodeIndex ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );
		if (isLeaf)
        {
			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;
            float minDistance = min(maxDistance, triangleDistance);
            bool found_intersection = intersectTriangles(bvh, rayOrigin, rayDirection, offset, count, minDistance,
				                                         faceIndices, faceNormal, barycoord, side, dist);
            if (found_intersection)
            {
                triangleDistance = minDistance;
                found = true;
            }
		}
        else
        {
			uint leftIndex = currNodeIndex + 1u;
			uint splitAxis = boundsInfo.x & 0x0000ffffu;
			uint rightIndex = boundsInfo.y;
			bool leftToRight = rayDirection[ splitAxis ] >= 0.0;
			uint c1 = leftToRight ? leftIndex : rightIndex;
			uint c2 = leftToRight ? rightIndex : leftIndex;
			// set c2 in the stack so we traverse it later. We need to keep track of a pointer in
			// the stack while we traverse. The second pointer added is the one that will be
			// traversed first
			ptr ++;
			stack[ ptr ] = c2;
			ptr ++;
			stack[ ptr ] = c1;
		}
	}
	return found;
}

bool trace(in vec3 rayOrigin, in vec3 rayDir, in float maxDistance,
            out vec3 P, out vec3 Ns, out vec3 Ng, out vec3 Ts, out vec3 baryCoord, out int material)
{
    // hit results
    uvec4 faceIndices_surface = uvec4(0u);
    vec3   faceNormal_surface = vec3(0.0, 0.0, 1.0);
    vec3    barycoord_surface = vec3(0.0);
    float        side_surface = 1.0;
    float        dist_surface = HUGE_DIST;
    bool hit_surface = bvhIntersectFirstHitWithinDistance( bvh_surface, rayOrigin, rayDir, maxDistance,
                                                           faceIndices_surface, faceNormal_surface, barycoord_surface, side_surface, dist_surface );
    uvec4 faceIndices_props = uvec4(0u);
    vec3   faceNormal_props = vec3(0.0, 0.0, 1.0);
    vec3    barycoord_props = vec3(0.0);
    float        side_props = 1.0;
    float        dist_props = HUGE_DIST;
    bool hit_props = bvhIntersectFirstHitWithinDistance( bvh_props, rayOrigin, rayDir, min(dist_surface, maxDistance),
                                                         faceIndices_props, faceNormal_props, barycoord_props, side_props, dist_props );

    bool hit = hit_surface || hit_props;
    if (!hit)
        return false;

    if (hit_surface && (!hit_props || (dist_surface <= dist_props)))
    {
        P = rayOrigin + dist_surface*rayDir;
        material = MATERIAL_OPENPBR;
        baryCoord = barycoord_surface;
        Ng = safe_normalize(faceNormal_surface);
        if (has_normals_surface)
        {
            Ns = textureSampleBarycoord(normalAttribute_surface, barycoord_surface, faceIndices_surface.xyz).xyz;
            const bool flip_normals = false;
            if (flip_normals)
                Ns *= -1.0;
        }
        else
            Ns = Ng;
        if (has_tangents_surface)
            Ts = textureSampleBarycoord(tangentAttribute_surface, barycoord_surface, faceIndices_surface.xyz).xyz;
        else
            Ts = normalToTangent(Ns);
    }

    else if (hit_props)
    {
        P = rayOrigin + dist_props*rayDir;
        material = MATERIAL_PROPS;
        baryCoord = barycoord_props;
        Ng = safe_normalize(faceNormal_props);
        if (has_normals_props)
        {
            Ns = textureSampleBarycoord(normalAttribute_props, barycoord_props, faceIndices_props.xyz).xyz;
            const bool flip_normals = false;
            if (flip_normals)
                Ns *= -1.0;
        }
        else
            Ns = Ng;
        if (has_tangents_props)
            Ts = textureSampleBarycoord(tangentAttribute_props, barycoord_props, faceIndices_props.xyz).xyz;
        else
            Ts = normalToTangent(Ns);
    }
    return true;
}


float TraceShadow(in vec3 rayOrigin, in vec3 rayDir, in float maxDistance)
{
    int material;
    vec3 pW, nsW, ngW, TsW, baryCoord;
    bool hit = trace(rayOrigin, rayDir, maxDistance,
                     pW, nsW, ngW, TsW, baryCoord, material);
    return hit ? 0.0 : 1.0;
}


////////////////////////////////////////////////
// "Neutral" color Lambertian BRDF for props
////////////////////////////////////////////////

vec3 neutral_brdf_evaluate(in vec3 pW, in Basis basis, in vec3 winputL, in vec3 woutputL,
                        inout float pdf_woutputL)
{
    if (winputL.z < DENOM_TOLERANCE || woutputL.z < DENOM_TOLERANCE) return vec3(0.0);
    pdf_woutputL = pdfHemisphereCosineWeighted(woutputL);
    if (wireframe && minComponent(basis.baryCoord) < 0.003) return vec3(0.0);
    return neutral_color / PI;
}

vec3 neutral_brdf_sample(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed,
                         out vec3 woutputL, out float pdf_woutputL)
{
    if (winputL.z < DENOM_TOLERANCE) return vec3(0.0);
    woutputL = sampleHemisphereCosineWeighted(rndSeed, pdf_woutputL);
    if (wireframe && minComponent(basis.baryCoord) < 0.003) return vec3(0.0);
    return neutral_color / PI;
}

//////////////////////////////////////
// BSDF dispatch
//////////////////////////////////////

vec3 evaluateBsdf(in vec3 pW, in Basis basis, in vec3 winputL, in vec3 woutputL, in int material,
                  inout float pdf_woutputL)
{
    if (material == MATERIAL_OPENPBR) return openpbr_bsdf_evaluate(pW, basis, winputL, woutputL, pdf_woutputL);
    else                              return neutral_brdf_evaluate(pW, basis, winputL, woutputL, pdf_woutputL);
}


vec3 sampleBsdf(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed, in int material,
                out vec3 woutputL, out float pdf_woutputL, out Volume internal_medium)
{
    if (material == MATERIAL_OPENPBR) return openpbr_bsdf_sample(pW, basis, winputL, rndSeed, woutputL, pdf_woutputL, internal_medium);
    else                              return neutral_brdf_sample(pW, basis, winputL, rndSeed, woutputL, pdf_woutputL);
}


/////////////////////////////////////////////////////////////////////////
// lighting
/////////////////////////////////////////////////////////////////////////

Basis sunBasis;

vec3 sunRadiance(in vec3 woutputW)
{
    float theta_max = sunAngularSize * PI/180.0;
    if (dot(woutputW, sunDir) < cos(theta_max)) return vec3(0.0);
    return sunPower * sunColor;
}

float sunTotalPower()
{
    return length(sunPower * sunColor);
}

vec3 sunSample(Basis basis,
               inout vec3 woutputL, inout vec3 woutputW, inout float pdfDir,
               inout uint rndSeed)
{
    float theta_max = sunAngularSize * PI/180.0;
    float theta = theta_max * sqrt(rand(rndSeed));
    float costheta = cos(theta);
    float sintheta = sqrt(max(0.0, 1.0-costheta*costheta));
    float phi = 2.0 * PI * rand(rndSeed);
    float cosphi = cos(phi);
    float sinphi = sin(phi);
    float x = sintheta * cosphi;
    float y = sintheta * sinphi;
    float z = costheta;
    float solid_angle = PI2 *(1.0 - cos(theta_max));
    pdfDir = 1.0/solid_angle;
    woutputW = localToWorld(vec3(x, y, z), sunBasis);
    woutputL = worldToLocal(woutputW, basis);
    // (Normalize by solid angle to make total power independent of angular size)
    return sunPower * sunColor / solid_angle;
}

float sunPdf(in vec3 woutputL, in vec3 woutputW)
{
    float theta_max = sunAngularSize * PI/180.0;
    if (dot(woutputW, sunDir) < cos(theta_max)) return 0.0;
    float solid_angle = 2.0*PI*(1.0 - cos(theta_max));
    return 1.0/solid_angle;
}

vec3 skyRadiance(in vec3 woutputW)
{
    vec4 env = textureLod(envMap, vec3(woutputW.x, woutputW.yz), 0.0);
    return env.rgb * skyPower * skyColor;
}

float skyTotalPower()
{
    return length(skyPower * skyColor) * PI2;
}

vec3 skySample(in Basis basis,
                out vec3 woutputL, out vec3 woutputW, out float pdfDir,
                inout uint rndSeed)
{
    woutputL = sampleHemisphereCosineWeighted(rndSeed, pdfDir);
    woutputW = localToWorld(woutputL, basis);
    return skyRadiance(woutputW);
}

float skyPdf(in vec3 woutputL, in vec3 woutputWs)
{
    return pdfHemisphereCosineWeighted(woutputL);
}

// Sample direct radiance at the given surface vertex
vec3 LiDirect(in vec3 pW, in Basis basis,
              out vec3 shadowL, out vec3 shadowW,
              out float lightPdf,
              inout uint rndSeed)
{
    // Do 1-sample MIS between uniform sky and sun sampling
    vec3 Li;
    {
        float w_sun = sunTotalPower();
        float w_sky = skyTotalPower();
        float P_sun = w_sun / (w_sun + w_sky);
        float P_sky = max(0.0, 1.0 - P_sun);
        float pdf_sun, pdf_sky;
        float r = rand(rndSeed);
        if (r < P_sun)
        {
            Li = sunSample(basis, shadowL, shadowW, pdf_sun, rndSeed);
            Li += skyRadiance(shadowW);
            pdf_sky = skyPdf(shadowL, shadowW);
        }
        else
        {
            Li = skySample(basis, shadowL, shadowW, pdf_sky, rndSeed);
            Li += sunRadiance(shadowW);
            pdf_sun = sunPdf(shadowL, shadowW);
        }
        lightPdf = P_sun*pdf_sun + P_sky*pdf_sky; // Light PDF according to 1-sample MIS
    }
    if (shadowL.z < 0.0) return vec3(0.0);
    if (maxComponent(Li) < RADIANCE_EPSILON) return vec3(0.0);
    float visibility = TraceShadow(pW, shadowW, HUGE_DIST);
    return visibility * Li;
}

// Corresponding PDF of direct radiance in the given shadow ray direction (for MIS)
float LiPDF(in vec3 shadowW, in Basis basis)
{
    vec3 shadowL = worldToLocal(shadowW, basis);
    float pdf_sky = skyPdf(shadowL, shadowW);
    float pdf_sun = sunPdf(shadowL, shadowW);
    float w_sun = sunTotalPower();
    float w_sky = skyTotalPower();
    float P_sun = w_sun / (w_sun + w_sky);
    float P_sky = max(0.0, 1.0 - P_sun);
    float lightPdf = P_sun*pdf_sun + P_sky*pdf_sky; // Light PDF according to 1-sample MIS
    return lightPdf;
}

vec3 evaluateEdf(in vec3 pW, in Basis basis, in vec3 winputL)
{
    return emission_color * emission_luminance;
}


/////////////////////////////////////////////////////////////////////////
// pathtracer
/////////////////////////////////////////////////////////////////////////

#define MIN_VOLUME_STEPS_BEFORE_RR 3

int sample_channel(in vec3 albedo, in vec3 throughput, inout uint rndSeed, inout vec3 channel_probs)
{
    // Sample color channel in proportion to throughput
    vec3 w = abs(throughput);
    float sum = w.r + w.g + w.b;
    channel_probs = w / max(DENOM_TOLERANCE, sum);
    float cdf = 0.0;
    float r = rand(rndSeed);
    for (int channel=0; channel<3; ++channel)
    {
        cdf += channel_probs[channel];
        if (r < cdf)
            return channel;
    }
    return 0;
}

#ifdef VOLUME_ENABLED
bool trace_volumetric(in vec3 pW, in vec3 dW, inout uint rndSeed,
                      in Volume volume,
                      out vec3 volume_throughput,
                      out vec3 pW_hit,
                      out vec3 dW_hit,
                      out vec3 NsW_hit,
                      out vec3 NgW_hit,
                      out vec3 TsW_hit,
                      out vec3 baryCoord_hit,
                      out int material_hit)
{
    // Do an "analogue random-walk" in the scattering medium, i.e. following the physical path of a photon.
    // Returns whether a surface hit occurred (and the hit data), and the volumetric path throughput.
    vec3 pWalk = pW;
    vec3 dWalk = dW;
    vec3 mfp = 1.0 / max(vec3(DENOM_TOLERANCE), volume.extinction);
    volume_throughput = vec3(1.0);
    for (int n=0; n < MAX_VOLUME_STEPS; ++n)
    {
        vec3 channel_probs;
        int channel = sample_channel(volume.albedo, volume_throughput, rndSeed, channel_probs);
        float walk_step = -log(rand(rndSeed)) * mfp[channel];
        bool surface_hit = trace(pWalk, dWalk, walk_step,
                                 pW_hit, NsW_hit, NgW_hit, TsW_hit, baryCoord_hit, material_hit);
        if (surface_hit)
        {
            // ray hits surface within walk_step, walk terminates.
            // update walk throughput on exit (via MIS)
            float dist_to_surface = length(pW_hit - pWalk);
            vec3 transmittance = exp(-dist_to_surface * volume.extinction);
            volume_throughput *= transmittance / max(DENOM_TOLERANCE, dot(channel_probs, transmittance));
            dW_hit = dWalk;
            return true;
        }
        // Scatter within the medium, and continue walking.
        // First, make a Russian-roulette termination decision (after a minimum number of steps has been taken)
        float termination_prob = 0.0;
        if (n > MIN_VOLUME_STEPS_BEFORE_RR)
        {
            float continuation_prob = clamp(maxComponent(volume_throughput), 0.0, 1.0);
            float termination_prob = 1.0 - continuation_prob;
            if (rand(rndSeed) < termination_prob)
                break;
            volume_throughput /= continuation_prob; // update walk throughput due to RR continuation
        }

        // update walk throughput on scattering in medium (via MIS)
        vec3 transmittance = exp(-walk_step * volume.extinction);
        volume_throughput *= volume.albedo * volume.extinction * transmittance;
        volume_throughput /= max(DENOM_TOLERANCE, dot(channel_probs, volume.extinction * transmittance));

        // walk in the sampled direction, staying inside the medium
        pWalk += walk_step * dWalk;

        // scatter into a new direction sampled from Henyey-Greenstein phase function
        dWalk = samplePhaseFunction(dWalk, volume.anisotropy, rndSeed);
        dWalk = normalize(dWalk);
    }
    volume_throughput = vec3(0.0); // path terminated in the medium
    return false;
}
#endif // VOLUME_ENABLED


void main()
{
    vec2 frag = gl_FragCoord.xy;

    // Initialize RNG
    uint rndSeed = uint(frag.x + frag.y*resolution.x);
    xorshift(rndSeed);
    rndSeed ^= uint(samples);

    // Apply FIS to obtain pixel jitter about center in pixel units
    const float filterRadius = 1.0;
    float jx = 0.5 * filterRadius * sample_triangle_filter(rand(rndSeed));
    float jy = 0.5 * filterRadius * sample_triangle_filter(rand(rndSeed));
    vec2 pixel = frag + vec2(jx, jy);

    // Get [-1, 1] normalized device coordinates,
    vec2 ndc = -1.0 + 2.0*(pixel/resolution.xy);

    // Compute primary camera ray in world-space
    vec3 pW, dW;
    ndcToCameraRay(ndc, invModelMatrix * cameraWorldMatrix, invProjectionMatrix,
                    pW, dW);
    dW = normalize(dW);

    // Setup sun basis
    sunBasis = makeBasis(sunDir);

    // Perform uni-directional pathtrace starting from the (pinhole) camera lens to estimate the primary ray radiance, L
    vec3 L = vec3(0.0);
    vec3 throughput = vec3(1.0);
    Basis basis;                      // kept for MIS book-keeping
    float bsdfPdf_continuation = 1.0; // ditto

#ifdef VOLUME_ENABLED
    // Initialize volumetric medium of camera ray
    // (NB, camera inside the volume is not handled properly in this implementation, for simplicity)
    Volume exterior_medium;
    exterior_medium.extinction = vec3(0.0);
    exterior_medium.albedo     = vec3(0.0);
    Volume current_medium = exterior_medium;
#endif
    bool in_dielectric = false;

#ifdef TRANSMISSION_ENABLED
    // Stochastically choose wavelength for potential dispersion effect
    // (here just a crude uniform sample of the visible range)
    bool dispersive = false;
    wavelength_nm = 360.0 + (700.0 - 360.0)*rand(rndSeed);
#endif // TRANSMISSION_ENABLED

    for (int vertex=0; vertex <= BOUNCES; vertex++)
    {
        // Generate next surface hit, given current vertex pW and current propagation direction dW
        // (where vertex 0 corresponds to the camera position)
        bool surface_hit;
        vec3 pW_next;
        vec3 NsW_next;
        vec3 NgW_next;
        vec3 TsW_next;
        vec3 baryCoord_next;
        int material_next;

        // Check for presence of volume
#ifdef VOLUME_ENABLED
        bool inside_volume            = in_dielectric && maxComponent(current_medium.extinction) > FLT_EPSILON;
        bool inside_scattering_volume = inside_volume && maxComponent(current_medium.albedo) > FLT_EPSILON;
#else
        bool inside_volume = false;
        bool inside_scattering_volume = false;
#endif

        // If not inside a scattering volume, ray proceeds in a straight line to the next surface hit
        if (!inside_scattering_volume)
        {
            // Raycast along current propagation direction dW, from current vertex pW
            surface_hit = trace(pW, dW, HUGE_DIST,
                                pW_next, NsW_next, NgW_next, TsW_next, baryCoord_next, material_next);

#ifdef VOLUME_ENABLED
            // Apply Beer-Lambert law for absorption
            if (surface_hit && inside_volume)
            {
                float ray_length = length(pW_next - pW);
                throughput *= exp(-ray_length * current_medium.extinction);
            }
#endif // VOLUME_ENABLED
        }

#ifdef VOLUME_ENABLED
        // Otherwise volumetric scattering may occur before the next surface hit
        else
        {
            vec3 volume_throughput;
            vec3 dW_next;
            surface_hit = trace_volumetric(pW, dW, rndSeed, current_medium, volume_throughput,
                                           pW_next, dW_next, NsW_next, NgW_next, TsW_next, baryCoord_next, material_next);
            dW = dW_next;
            throughput *= volume_throughput;
        }
#endif // VOLUME_ENABLED

        if (!surface_hit)
        {
            // Ray missed all geometry; add contribution from distant lights
            float misWeightLight = 1.0;
            if (vertex > 0)
            {
                float lightPdf = LiPDF(dW, basis); // surface basis of previous hit
                misWeightLight = powerHeuristic(bsdfPdf_continuation, lightPdf);
            }
            L += throughput * misWeightLight * (sunRadiance(dW) + skyRadiance(dW));
            break; // Ray escapes to infinity, terminate path
        }

        // Terminate at max bounce count (biased)
        if (vertex == BOUNCES)
            break;

        // Update to the next surface vertex.
        // First, compute the normal and thus the local vertex basis:
        pW             = pW_next;
        vec3 NsW       = NsW_next;
        vec3 NgW       = NgW_next;
        vec3 TsW       = TsW_next;
        vec3 baryCoord = baryCoord_next;
        int material   = material_next;

        if (material == MATERIAL_OPENPBR)
        {
            // Orient local shading normal so that it points from the surface interior to the exterior
            if ( (in_dielectric && dot(NsW, dW) < 0.0) ||
                (!in_dielectric && dot(NsW, dW) > 0.0))
                NsW *= -1.0;
        }
        else
        {
            // Otherwise surface is opaque, must be approaching from the exterior
            if (dot(NsW, dW) > 0.0)
                NsW *= -1.0;
        }

        // Align geometric normal into same hemisphere as shading normal
        if (dot(NgW, NsW) < 0.0) NgW *= -1.0;

        // Construct local shading frame
        if (smooth_normals)
        {
            // If the surface is opaque, but the incident ray lies below the hemisphere of the normal,
            // which can occur due to shading normals, apply the "Flipping hack" to prevent artifacts
            // (see Schüßler, "Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing")
            if (material == MATERIAL_OPENPBR && openpbr_is_opaque() && dot(NsW, dW) > 0.0)
                NsW = 2.0*NgW*dot(NgW, NsW) - NsW;
            basis = makeBasis(NsW, TsW_next, baryCoord);
        }
        else
            basis = makeBasis(NgW, TsW_next, baryCoord);

        vec3 winputW = -dW; // winputW, points *towards* the incident direction (parallel to photon)
        vec3 winputL = worldToLocal(winputW, basis);

        // Prepare OpenPBR if that material is used at the current vertex
        if (material == MATERIAL_OPENPBR)
            openpbr_prepare(pW, basis, winputL, rndSeed);

        // Sample BSDF for the continuation ray direction
        Volume internal_medium;
        vec3 surface_throughput;
        {
            vec3 woutputL; // points *towards* the outgoing ray direction (opposite to photon)
            vec3 f = sampleBsdf(pW, basis, winputL, rndSeed, material, woutputL, bsdfPdf_continuation, internal_medium);
            vec3 woutputW = localToWorld(woutputL, basis);
            surface_throughput = f / max(PDF_EPSILON, bsdfPdf_continuation) * abs(dot(woutputW, basis.nW));
            dW = woutputW; // Update continuation ray direction to the BSDF-sampled direction
        }

        // Add emission from the surface point, if present
        if (material == MATERIAL_OPENPBR)
            L += throughput * evaluateEdf(pW, basis, winputL);

        // Prepare for tracing the direct lighting and continuation rays
        pW += NgW * sign(dot(dW, NgW)) * RAY_OFFSET; // perturb vertex into geometric half-space of scattered ray

        // Check if a transmission has occurred, and update the current_medium and dispersion state accordingly.
        bool transmitted = (material == MATERIAL_OPENPBR) && (dot(winputW, NgW) * dot(dW, NgW) < 0.0);
        if (transmitted)
        {
#ifdef TRANSMISSION_ENABLED
            if (!in_dielectric && !dispersive)
            {
                // On first transmission into dielectric, apply associated color of stochastically chosen wavelength
                // (where the color channel normalization here is just an approximation)
                if (transmission_dispersion_scale > 0.0)
                    surface_throughput *= xyzToRgb(xyzFit_1931(wavelength_nm)) * vec3(2.7, 3.3, 3.45);
                dispersive = true;
            }
#endif // DISPERSION_ENABLED

            // Update in_dielectric state
            in_dielectric = !in_dielectric;

#ifdef VOLUME_ENABLED
            // Thus update current medium
            if (in_dielectric)
                current_medium = internal_medium;
            else
                current_medium = exterior_medium;
#endif // VOLUME_ENABLED
        } // transmitted

        // Add direct lighting term at the current surface vertex
        if (!in_dielectric && !transmitted)
        {
            vec3 shadowL, shadowW; // sampled shadow ray direction
            float lightPdf;
            vec3 Li = LiDirect(pW, basis, shadowL, shadowW, lightPdf, rndSeed);
            if (maxComponent(Li) > RADIANCE_EPSILON)
            {
                float bsdfPdf_shadow = PDF_EPSILON;
                vec3 fshadow = evaluateBsdf(pW, basis, winputL, shadowL, material, bsdfPdf_shadow);
                float misWeightLight = powerHeuristic(lightPdf, bsdfPdf_shadow);
                L += throughput * misWeightLight * fshadow * abs(dot(shadowW, basis.nW)) * Li / max(PDF_EPSILON, lightPdf);
            }
        } // direct lighting

        // Update path continuation throughput
        throughput *= surface_throughput;

        // Russian roulette termination (unbiased)
        if (maxComponent(throughput) < 1.0 && vertex > 1)
        {
            float q = max(0.0, 1.0 - maxComponent(throughput));
            if (rand(rndSeed) < q)
                break;
            throughput /= 1.0 - q;
        }

    } // bounce loop

    gl_FragColor.rgb = L;
    gl_FragColor.a = accumulation_weight; // Implements Monte-Carlo accumulation via alpha blending
}
