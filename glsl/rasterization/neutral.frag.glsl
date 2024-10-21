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

uniform vec3  neutral_color;

//////////////////////////////////////////////////////
// lighting uniforms
//////////////////////////////////////////////////////

uniform float skyPower;
uniform vec3 skyColor;

uniform float sunPower;
uniform vec3 sunColor;
uniform vec3 sunDir;
varying vec3 vSunDir;


//////////////////////////////////////////////////////
// useful constants
//////////////////////////////////////////////////////

const float PI                    = 3.141592653589793;

// tolerances
const float DENOM_TOLERANCE       = 1.0e-10;

struct Basis
{
    // right-handed coordinate system
    vec3 nW; // aligned with the z-axis in local space
    vec3 tW; // aligned with the x-axis in local space
    vec3 bW; // aligned with the y-axis in local space
};

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


///////////////////////////////////////////////////////////////////////
// Diffuse BRDF
///////////////////////////////////////////////////////////////////////

vec3 diffuse_brdf_evaluate(in vec3 V, in vec3 L)
{
    return neutral_color / PI;
}

vec3 diffuse_brdf_albedo(in vec3 V)
{
    return neutral_color;
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

    // Set up vertex frame (in view space, following THREE's convention)
    vec3 Nworld = normalize(inverseTransformDirection(normal, viewMatrix));
    Basis basis = makeBasis(Nworld);

    vec3 Vworld = normalize(cameraPosition - vWorldPosition);
    vec3 Vlocal = worldToLocal(Vworld, basis);

    // Compute direct lighting
    vec3 radiance = vec3(0.0);
    {
        // Sun contribution
        {
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
                    sun_visibility *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
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
                vec3 f = diffuse_brdf_evaluate(Vlocal, Llocal);
                radiance += sun_visibility * f * abs(NdotL) * sunColor * sunPower;
            }
        }
        // Sky contribution
        {
            // diffuse contribution
            {
                vec4 diff_env = textureLod(envMap, vec3(0.0, 1.0, 0.0), 1000.0);
                radiance += skyColor * skyPower * diff_env.rgb * diffuse_brdf_albedo(Vlocal);
            }
        }
    }

    gl_FragColor.rgb = radiance;
    gl_FragColor.a = 1.0;

}
