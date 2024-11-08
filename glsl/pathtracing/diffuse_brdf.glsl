
/* Implements:

    https://arxiv.org/abs/2410.18026

    "EON: A practical energy-preserving rough diffuse BRDF",
        Jamie Portsmouth, Peter Kutz, Stephen Hill
*/

const float constant1_FON = 0.5f - 2.0f / (3.0f * PI);
const float constant2_FON = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// FON directional albedo (analytic)
float E_FON_exact(float mu, float r)
{
    #define safe_acos(x) (acos(clamp(x, -1.0, 1.0)))
    float AF = 1.0f / (1.0f + constant1_FON * r); // FON A coeff.
    float BF = r * AF;                            // FON B coeff.
    float Si = sqrt(1.0f - (mu * mu));
    float G = Si * (acos(clamp(mu, -1.0f, 1.0f)) - Si * mu)
            + (2.0f / 3.0f) * ((Si / mu) * (1.0f - (Si * Si * Si)) - Si);
    return AF + (BF/PI) * G;
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

////////////////////////////////////////////////////////////////////////////////
// EON BRDF
////////////////////////////////////////////////////////////////////////////////

  // Evaluates EON BRDF value, given inputs:
  //       rho = single-scattering albedo parameter
  //         r = roughness in [0, 1]
  //  wi_local = direction of incident ray (directed away from vertex)
  //  wo_local = direction of outgoing ray (directed away from vertex)
  //     exact = flag to select exact or fast approx. version
  //
  vec3 f_EON(vec3 rho, float r, vec3 wi_local, vec3 wo_local, bool exact)
  {
      float mu_i = wi_local.z;                               // input angle cos
      float mu_o = wo_local.z;                               // output angle cos
      float s = dot(wi_local, wo_local) - mu_i * mu_o;       // QON $s$ term
      float sovertF = s > 0.0f ? s / max(mu_i, mu_o) : s;    // FON $s/t$
      float AF = 1.0f / (1.0f + constant1_FON * r);          // FON $A$ coeff.
      vec3 f_ss = (rho/PI) * AF * (1.0f + r * sovertF); // single-scatter lobe
      float EFo = exact ? E_FON_exact(mu_o, r):              // FON $w_o$ albedo (exact)
                          E_FON_approx(mu_o, r);             // FON $w_o$ albedo (approx)
      float EFi = exact ? E_FON_exact(mu_i, r):              // FON $w_i$ albedo (exact)
                          E_FON_approx(mu_i, r);             // FON $w_i$ albedo (approx)
      float avgEF = AF * (1.0f + constant2_FON * r);         // avg. albedo
      vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0f) - rho * (1.0f - avgEF));
      const float eps = 1.0e-7f;
      vec3 f_ms = (rho_ms/PI) * max(eps, 1.0f - EFo)    // multi-scatter lobe
                              * max(eps, 1.0f - EFi)
                              / max(eps, 1.0f - avgEF);
      return f_ss + f_ms;
  }

////////////////////////////////////////////////////////////////////////////////
// EON directional albedo
////////////////////////////////////////////////////////////////////////////////

  vec3 E_EON(vec3 rho, float r, vec3 wi_local, bool exact)
  {
      float mu_i = wi_local.z;                       // input angle cos
      float AF = 1.0f / (1.0f + constant1_FON * r);  // FON $A$ coeff.
      float EF = exact ? E_FON_exact(mu_i, r):       // FON $w_i$ albedo (exact)
                         E_FON_approx(mu_i, r);      // FON $w_i$ albedo (approx)
      float avgEF = AF * (1.0f + constant2_FON * r); // average albedo
      vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0f) - rho * (1.0f - avgEF));
      return rho * EF + rho_ms * (1.0f - EF);
  }

////////////////////////////////////////////////////////////////////////////////
// EON importance sampling
////////////////////////////////////////////////////////////////////////////////

void ltc_terms(float mu, float r,
               out float a, out float b, out float c, out float d)
{
    a = 1.0 + r*(0.303392f + (-0.518982f + 0.111709f*mu)*mu + (-0.276266f + 0.335918f*mu)*r);
    b = r*(-1.16407f + 1.15859f*mu + (0.150815f - 0.150105f*mu)*r)/(mu*mu*mu - 1.43545f);
    c = 1.0f + (0.20013f + (-0.506373f + 0.261777f*mu)*mu)*r;
    d = ((0.540852f + (-1.01625f + 0.475392f*mu)*mu)*r)/(-1.0743f + mu*(0.0725628f + mu));
}

vec4 cltc_sample(in vec3 wo_local, float r, float u1, float u2)
{
    float a, b, c, d; ltc_terms(wo_local.z, r, a, b, c, d);    // coeffs of LTC $M$
    float R = sqrt(u1); float phi = 2.0f * PI * u2;            // CLTC sampling
    float x = R * cos(phi); float y = R * sin(phi);            // CLTC sampling
    float vz = 1.0f / sqrt(d*d + 1.0f);                        // CLTC sampling factors
    float s = 0.5f * (1.0f + vz);                              // CLTC sampling factors
    x = -mix(sqrt(1.0f - y*y), x, s);                          // CLTC sampling
    vec3 wh = vec3(x, y, sqrt(max(1.0f - (x*x + y*y), 0.0f))); // $w_h$ sample via CLTC
    float pdf_wh = wh.z / (PI * s);                            // PDF of $w_h$ sample
    vec3 wi = vec3(a*wh.x + b*wh.z, c*wh.y, d*wh.x + wh.z);    // $M w_h$ (unnormalized)
    float len = length(wi);                                    // $|M w_h| = 1/|M^{-1} w_h|$
    float detM = c*(a - b*d);                                  // $|M|$
    float pdf_wi = pdf_wh * len*len*len / detM;                // $w_i$ sample PDF
    mat3 fromLTC = orthonormal_basis_ltc(wo_local);            // transform $w_i$ to world space
    wi = normalize(fromLTC * wi);                              // transform $w_i$ to world space
    return vec4(wi, pdf_wi);
}

float cltc_pdf(in vec3 wo_local, in vec3 wi_local, float r)
{
    mat3 toLTC = transpose(orthonormal_basis_ltc(wo_local));                 // transform $w_i$ to LTC space
    vec3 wi = toLTC * wi_local;                                              // transform $w_i$ to LTC space
    float a, b, c, d; ltc_terms(wo_local.z, r, a, b, c, d);                  // coeffs of LTC $M$
    float detM = c*(a - b*d);                                                // $|M|$
    vec3 wh = vec3(c*(wi.x - b*wi.z), (a - b*d)*wi.y, -c*(d*wi.x - a*wi.z)); // $\mathrm{adj}(M) wi$
    float lensq = dot(wh, wh);                                               // $|M| |M^{-1} wi|$
    float vz = 1.0f / sqrt(d*d + 1.0f);                                      // CLTC sampling factors
    float s = 0.5f * (1.0f + vz);                                            // CLTC sampling factors
    float pdf = sqr(detM / lensq) * max(wh.z, 0.0f) / (PI * s);              // $w_i$ sample PDF
    return pdf;
}

vec3 uniform_lobe_sample(float u1, float u2)
{
    float sinTheta = sqrt(1.0f - u1*u1);
    float phi = 2.0f * PI * u2;
    return vec3(sinTheta * cos(phi), sinTheta * sin(phi), u1);
}

vec4 sample_EON(in vec3 wo_local, float r, float u1, float u2)
{
    float mu = wo_local.z;
    float P_u = pow(r, 0.1f) * (0.162925f + mu*(-0.372058f + (0.538233f - 0.290822f*mu)*mu));
    float P_c = 1.0f - P_u;
    vec4 wi; float pdf_c;
    if (u1 <= P_u) {
        u1 = u1 / P_u;
        wi.xyz = uniform_lobe_sample(u1, u2);
        pdf_c = cltc_pdf(wo_local, wi.xyz, r); }
    else {
        u1 = (u1 - P_u) / P_c;
        wi = cltc_sample(wo_local, r, u1, u2);
        pdf_c = wi.w; }
    const float pdf_u = 1.0f / (2.0f * PI);
    wi.w = P_u*pdf_u + P_c*pdf_c;
    return wi;
}

float pdf_EON(in vec3 wo_local, in vec3 wi_local, float r)
{
    float mu = wo_local.z;
    float P_u = pow(r, 0.1f) * (0.162925f + mu*(-0.372058f + (0.538233f - 0.290822f*mu)*mu));
    float P_c = 1.0 - P_u;
    float pdf_c = cltc_pdf(wo_local, wi_local, r);
    const float pdf_u = 1.0f / (2.0f * PI);
    return P_u*pdf_u + P_c*pdf_c;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// diffuse BRDF
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 diffuse_brdf_evaluate(in vec3 pW, in Basis basis, in vec3 winputL, in vec3 woutputL,
                           inout float pdf_woutputL)
{
    if (winputL.z < DENOM_TOLERANCE || woutputL.z < DENOM_TOLERANCE) return vec3(0.0f);
    pdf_woutputL = pdf_EON(winputL, woutputL, base_roughness);
    return f_EON(base_weight * base_color, base_roughness, winputL, woutputL, true);
}

vec3 diffuse_brdf_sample(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed,
                         out vec3 woutputL, out float pdf_woutputL)
{
    if (winputL.z < DENOM_TOLERANCE) return vec3(0.0f);
    float u1 = rand(rndSeed); float u2 = rand(rndSeed);
    vec4 wiP = sample_EON(winputL, base_roughness, u1, u2);
    woutputL     = wiP.xyz;
    pdf_woutputL = wiP.w;
    return f_EON(base_weight * base_color, base_roughness, winputL, woutputL, true);
}

vec3 diffuse_brdf_albedo(in vec3 pW, in Basis basis, in vec3 winputL, inout uint rndSeed)
{
    if (winputL.z < DENOM_TOLERANCE) return vec3(0.0f);
    return E_EON(base_weight * base_color, base_roughness, winputL, true);
}
