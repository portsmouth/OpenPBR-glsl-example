
////////////////////////////////////////////////////////////////////////////////////
// EON model
////////////////////////////////////////////////////////////////////////////////////

#define constant1_FON (0.5 - 2.0/(3.0*PI))
#define constant2_FON (2.0/3.0 - 28.0/(15.0*PI))

// FON directional albedo (analytic)
float E_FON_analyt(float mu, float sigma)
{
    #define safe_acos(x) (acos(clamp(x, -1.0, 1.0)))
    float AF = 1.0 / (1.0 + constant1_FON*sigma); // FON model A coefficient
    float BF = sigma * AF;                        // FON model B coefficient
    float Si = max(0.0, sqrt(1.0 - mu * mu));
    float G = Si * (safe_acos(mu) - Si*mu) + 2.0*((Si/mu)*(1.0 - pow(Si, 3.0)) - Si)/3.0;
    float E = AF + (BF/PI)*G;
    return E;
}

// FON directional albedo (approx.)
float E_FON_approx(float mu, float r)
{
    float mucomp = 1.0f - mu;
    float mucomp2 = mucomp * mucomp;
    const mat2 Gcoeffs = mat2(0.0571085289f, -0.332181442f,
    0.491881867f, 0.0714429953f);
    float GoverPi = dot(Gcoeffs * vec2(mucomp, mucomp2), vec2(1.0f, mucomp2));
    return (1.0f + r * GoverPi) / (1.0f + constant1_FON * r);
}

// EON BRDF
vec3 f_EON(in vec3 rho, float sigma, in vec3 wi, in vec3 wo, bool exact)
{
    float muI = wi.z;                                    // input angle cos
    float muO = wo.z;                                    // output angle cos
    float s = dot(wi, wo) - muI * muO;                   // QON s term
    float stinv = s > 0.0 ? s / max(muI, muO) : s;       // Fujii model stinv
    float AF = 1.0 / (1.0 + constant1_FON*sigma);        // Fujii model A coefficient
    vec3 f_ss = (rho / PI) * AF * (1.0 + sigma*stinv);   // single-scatt. BRDF
    float EFo = exact ? E_FON_analyt(muO, sigma) :       // EFo at rho=1 (analytic)
                        E_FON_approx(muO, sigma);        // EFo at rho=1 (approx)
    float EFi = exact ? E_FON_analyt(muI, sigma) :       // EFi at rho=1 (analytic)
                        E_FON_approx(muI, sigma);        // EFi at rho=1 (approx)
    float avgEF = AF * (1.0 + constant2_FON*sigma);      // avg. albedo
    vec3 rho_ms = sqr(rho) * avgEF / (vec3(1.0) - rho*max(0.0, 1.0-avgEF));
    vec3 f_ms = (rho_ms / PI) * max(1e-7, 1.0 - EFo) *   // multi-scatter lobe
                                max(1e-7, 1.0 - EFi) /
                                max(1e-7, 1.0 - avgEF);
    return f_ss + f_ms;
}

// EON albedo
vec3 E_EON(in vec3 rho, float sigma, in vec3 wi)
{
    float muI = wi.z;                                    // input angle cos
    float AF = 1.0 / (1.0 + constant1_FON*sigma);        // FON model A coefficient
    float EF = E_FON_approx(muI, sigma);                 // EFi at rho=1 (approx)
    float avgEF = AF * (1.0 + constant2_FON*sigma);      // average albedo
    vec3 rho_ms = sqr(rho) * avgEF / (vec3(1.0) - rho*max(0.0, 1.0-avgEF));
    return rho*EF + rho_ms*(1.0 - EF);
}

void ltc_terms(float mu, float r,
               out float a, out float b, out float c, out float d)
{
    a = 1.0 + r*(0.303392 + (-0.518982 + 0.111709*mu)*mu + (-0.276266 + 0.335918*mu)*r);
    b = r*(-1.16407 + 1.15859*mu + (0.150815 - 0.150105*mu)*r)/(mu*mu*mu - 1.43545);
    c = 1.0 + (0.20013 + (-0.506373 + 0.261777*mu)*mu)*r;
    d = ((0.540852 + (-1.01625 + 0.475392*mu)*mu)*r)/(-1.0743 + mu*(0.0725628 + mu));
}

// (NB, uses opposite convention that woutputL is the incident/camera ray direction)
vec4 cltc_sample(in vec3 woutputL, float r, float u1, float u2)
{
    float a, b, c, d; ltc_terms(woutputL.z, r, a, b, c, d);  // coeffs of LTC $M$
    float R = sqrt(u1); float phi = 2.0 * PI * u2;           // CLTC sampling
    float x = R * cos(phi); float y = R * sin(phi);          // CLTC sampling
    float vz = 1.0 / sqrt(d*d + 1.0);                        // CLTC sampling factors
    float s = 0.5 * (1.0 + vz);                              // CLTC sampling factors
    x = -mix(sqrt(1.0 - y*y), x, s);                         // CLTC sampling
    vec3 wh = vec3(x, y, sqrt(max(1.0 - (x*x + y*y), 0.0))); // $w_h$ sample via CLTC
    float pdf_wh = wh.z / (PI * s);                          // PDF of $w_h$ sample
    vec3 wi = vec3(a*wh.x + b*wh.z, c*wh.y, d*wh.x + wh.z);  // $M w_h$ (unnormalized)
    float len = length(wi);                                  // $|M w_h| = 1/|M^{-1} w_h|$
    float detM = c*(a - b*d);                                // $|M|$
    float pdf_wi = pdf_wh * len*len*len / detM;              // $w_i$ sample PDF
    mat3 fromLTC = orthonormal_basis_ltc(woutputL);          // transform $w_i$ to world space
    wi = normalize(fromLTC * wi);                            // transform $w_i$ to world space
    return vec4(wi, pdf_wi);
}

// (NB, uses opposite convention that woutputL is the incident/camera ray direction)
float cltc_pdf(in vec3 woutputL, in vec3 winputL, float r)
{
    mat3 toLTC = transpose(orthonormal_basis_ltc(woutputL));                 // transform $w_i$ to LTC space
    vec3 wi = toLTC * winputL;                                               // transform $w_i$ to LTC space
    float a, b, c, d; ltc_terms(woutputL.z, r, a, b, c, d);                  // coeffs of LTC $M$
    float detM = c*(a - b*d);                                                // $|M|$
    vec3 wh = vec3(c*(wi.x - b*wi.z), (a - b*d)*wi.y, -c*(d*wi.x - a*wi.z)); // $\mathrm{adj}(M) wi$
    float lensq = dot(wh, wh);                                               // $|M| |M^{-1} wi|$
    float vz = 1.0 / sqrt(d*d + 1.0);                                        // CLTC sampling factors
    float s = 0.5 * (1.0 + vz);                                              // CLTC sampling factors
    float pdf = sqr(detM / lensq) * max(wh.z, 0.0) / (PI * s);               // $w_i$ sample PDF
    return pdf;
}

vec3 uniform_lobe_sample(float u1, float u2)
{
    float z = u1;
    float R = sqrt(1.0 - z*z); float phi = 2.0 * PI * u2;
    float x = R * cos(phi); float y = R * sin(phi);
    return vec3(x, y, z);
}

// (NB, uses opposite convention that woutputL is the incident/camera ray direction)
vec4 sample_EON(in vec3 woutputL, float r, float u1, float u2)
{
    float mu = woutputL.z;
    float P_u = pow(r, 0.1) * (0.162925 + mu*(-0.372058 + (0.538233 - 0.290822*mu)*mu));
    float P_c = 1.0 - P_u;
    vec4 wi; float pdf_C;
    if (u1 <= P_u) {
        u1 = u1 / P_u;
        wi.xyz = uniform_lobe_sample(u1, u2);
        pdf_C = cltc_pdf(woutputL, wi.xyz, r); }
    else {
        u1 = (u1 - P_u) / P_c;
        wi = cltc_sample(woutputL, r, u1, u2);
        pdf_C = wi.w; }
    const float pdf_U = 1.0 / (2.0 * PI);
    wi.w = P_u*pdf_U + P_c*pdf_C;
    return wi;
}

// (NB, uses opposite convention that woutputL is the incident/camera ray direction)
float pdf_EON(in vec3 woutputL, in vec3 winputL, float r)
{
    float mu = woutputL.z;
    float P_u = pow(r, 0.1) * (0.162925 + mu*(-0.372058 + (0.538233 - 0.290822*mu)*mu));
    float P_c = 1.0 - P_u;
    float pdf_cltc = cltc_pdf(woutputL, winputL, r);
    const float pdf_U = 1.0 / (2.0 * PI);
    return P_u*pdf_U + P_c*pdf_cltc;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diffuse BRDF
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 diffuse_brdf_evaluate(in vec3 pW, in Basis basis, in vec3 winputL, in vec3 woutputL,
                           inout float pdf_woutputL)
{
    if (winputL.z < DENOM_TOLERANCE || woutputL.z < DENOM_TOLERANCE) return vec3(0.0);
    pdf_woutputL = pdf_EON(winputL, woutputL, base_roughness);
    return f_EON(base_weight * base_color, base_roughness, winputL, woutputL, true);
}

vec3 diffuse_brdf_sample(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed,
                         out vec3 woutputL, out float pdf_woutputL)
{
    if (winputL.z < DENOM_TOLERANCE) return vec3(0.0);
    float u1 = rand(rndSeed); float u2 = rand(rndSeed);
    vec4 woP = sample_EON(winputL, base_roughness, u1, u2);
    woutputL     = woP.xyz;
    pdf_woutputL = woP.w;
    return f_EON(base_weight * base_color, base_roughness, winputL, woutputL, true);
}

vec3 diffuse_brdf_albedo(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed)
{
    // Estimate of the BRDF albedo, used to compute the discrete probability of selecting this lobe
    if (winputL.z < DENOM_TOLERANCE) return vec3(0.0);
    return E_EON(base_weight * base_color, base_roughness, winputL);
}
