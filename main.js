

import { Scene,
    Vector2, Vector3, Matrix4, Box3, Color,
    Mesh, MeshBasicMaterial, MeshStandardMaterial, MeshLambertMaterial, ShaderMaterial,
    PerspectiveCamera, OrthographicCamera,
    DirectionalLight, AmbientLight, DoubleSide,
    LinearSRGBColorSpace, SRGBColorSpace, RGBAFormat, FloatType,
    WebGLRenderer, WebGLRenderTarget, TextureLoader,
    EquirectangularReflectionMapping, CubeReflectionMapping,
    UniformsUtils, UniformsLib, ShaderLib,
    PCFSoftShadowMap, CameraHelper  } from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import Stats from 'stats.js';

import {
MeshBVH, MeshBVHUniformStruct, FloatVertexAttributeTexture,
shaderStructs, shaderIntersectFunction, SAH, StaticGeometryGenerator
} from 'three-mesh-bvh';

import { GUI } from 'lil-gui';

import glsl_pathtracing_main            from './glsl/pathtracing/main.glsl?raw'
import glsl_pathtracing_fuzz_brdf       from './glsl/pathtracing/fuzz_brdf.glsl?raw'
import glsl_pathtracing_coat_brdf       from './glsl/pathtracing/coat_brdf.glsl?raw'
import glsl_pathtracing_thin_film       from './glsl/pathtracing/thin-film.glsl?raw'
import glsl_pathtracing_metal_brdf      from './glsl/pathtracing/metal_brdf.glsl?raw'
import glsl_pathtracing_specular_brdf   from './glsl/pathtracing/specular_brdf.glsl?raw'
import glsl_pathtracing_specular_btdf   from './glsl/pathtracing/specular_btdf.glsl?raw'
import glsl_pathtracing_diffuse_brdf    from './glsl/pathtracing/diffuse_brdf.glsl?raw'
import glsl_pathtracing_openpbr_surface from './glsl/pathtracing/openpbr_surface.glsl?raw'
import glsl_pathtracing_pathtracer      from './glsl/pathtracing/pathtracer.glsl?raw'

import glsl_rasterization_openpbr_frag          from './glsl/rasterization/openpbr.frag.glsl?raw'
import glsl_rasterization_openpbr_vert          from './glsl/rasterization/openpbr.vert.glsl?raw'

import glsl_rasterization_neutral_frag          from './glsl/rasterization/neutral.frag.glsl?raw'
import glsl_rasterization_neutral_vert          from './glsl/rasterization/neutral.vert.glsl?raw'

import { Circle } from 'progressbar.js'

class MeshLoader
{
    constructor()
    {
        this.result = null;
        this.loader = new GLTFLoader();
    }

    reset()
    {
        this.result = null;
    }

    async load(path)
    {
        if (this.result) Promise.resolve(this.result);

        let gltf = await this.loader.loadAsync(path);
        let S = Array.isArray( gltf.scene ) ? gltf.scene : [ gltf.scene ];
        const meshes = [];

        for ( let i = 0, l = S.length; i < l; i++ )
        {
            S[i].traverseVisible( c =>
                {
                    if (c.isMesh)
                    {
                        meshes.push(c);
                    }
                }
            )
        }

        if (meshes.length > 0)
        {
            const generator = new StaticGeometryGenerator(meshes);
            generator.attributes = [ 'position', 'color', 'normal', 'tangent', 'uv', 'uv2' ];
            generator.applyWorldTransforms = false;
            let merged_mesh = new Mesh(generator.generate(), new MeshStandardMaterial());

            let bvh = new MeshBVH( merged_mesh.geometry, { strategy: SAH, maxLeafTris: 1 } );
            this.result = {scene:gltf.scene, bvh:bvh, mesh:merged_mesh};
            console.log("==> loaded mesh ", path);
        }

        return this.result;
    }
}

function array_to_vector3(array)
{
    return new Vector3(array[0], array[1], array[2]);
}

var params =
{
    //////////////////////////////////////////////////////
    // renderer params
    //////////////////////////////////////////////////////

    scene_name:                         'standard-shader-ball',
    smooth_normals:                     true,
    bounces:                            10,
    max_samples:                        1024,
    max_volume_steps:                   8,
    wireframe:                          false,
    neutral_color:                      [0.99, 0.99, 0.99],

    //////////////////////////////////////////////////////
    // lighting params
    //////////////////////////////////////////////////////

    skyPower:                            1.0,
    skyColor:                            [1.0, 1.0, 1.0],
    sunPower:                            0.25,
    sunAngularSize:                      5.0,
    sunLatitude:                         40.0,
    sunLongitude:                        315.0,
    sunColor:                            [1.0, 1.0, 1.0],

    //////////////////////////////////////////////////////
    // OpenPBR surface params
    //////////////////////////////////////////////////////

    base_weight:                         1.0,
    base_color:                          [0.8, 0.8, 0.8],
    base_roughness:                      0.0,
    base_metalness:                      0.0,

    specular_weight:                     1.0,
    specular_color:                      [1.0, 1.0, 1.0],
    specular_roughness:                  0.1,
    specular_anisotropy:                 0.0,
    specular_rotation:                   0.0,
    specular_ior:                        1.5,

    transmission_weight:                 0.0,
    transmission_color:                  [1.0, 1.0, 1.0],
    transmission_depth:                  0.0,
    transmission_scatter:                [0.0, 0.0, 0.0],
    transmission_scatter_anisotropy:     0.0,
    transmission_dispersion_abbe_number: 20.0,
    transmission_dispersion_scale:       0.0,

    subsurface_weight:                   0.0,
    subsurface_color:                    [0.8, 0.8, 0.8],
    subsurface_radius:                   0.2,
    subsurface_radius_scale:             [1.0, 0.5, 0.25],
    subsurface_anisotropy:               0.0,

    coat_weight:                         0.0,
    coat_color:                          [1.0, 1.0, 1.0],
    coat_roughness:                      0.0,
    coat_anisotropy:                     0.0,
    coat_rotation:                       0.0,
    coat_ior:                            1.6,
    coat_darkening:                      1.0,

    fuzz_weight:                         0.0,
    fuzz_color:                          [1.0, 1.0, 1.0],
    fuzz_roughness:                      0.5,

    emission_luminance:                  0.0,
    emission_color:                      [1.0, 1.0, 1.0],

    thin_film_weight:                    0.0,
    thin_film_thickness:                 1000.0,
    thin_film_ior:                       1.4,

    geometry_opacity:                    1.0,
    geometry_thin_walled:                false,

    reset_camera:                        function() { reset_camera(params.scene_name); }

};

var materialDefines = {
    BOUNCES:              params.bounces,
    MAX_VOLUME_STEPS:     params.max_volume_steps,

    FUZZ_ENABLED:         false,
    COAT_ENABLED:         false,
    TRANSMISSION_ENABLED: false,
    VOLUME_ENABLED:       false,
    DISPERSION_ENABLED:   false,
    THIN_FILM_ENABLED:    false
};

var mesh_loader;
var renderer, camera, orbitControls, scene, gui, stats;
var pathtracedQuad, pathtracedFinalQuad, pathtracingRenderTarget;
var pathtracedMaterial = null;
var openpbrMaterial = null;
var neutralMaterial = null;
var materialDefines;
var directionalLight, ambientLight;
var camera_initialized = false;
var env_map_texture = null;

var MESH_SURFACE;
var MESH_PROPS;
var BVH_SURFACE;
var BVH_PROPS;

var progress_bar;
var progress_finished_timer;

var LOADED;
var COMPILING;
var PATHTRACING;
var samples = 0;

var scene_names = {
    'Standard Shader Ball': 'standard-shader-ball',
    'Glavenus':             'glavenus',
    'Terrain':              'terrain',
    'Bearded Man':          'bearded-man'
}

init();
render();

function updateSunDir()
{
    let latTheta = (90.0-params.sunLatitude) * Math.PI/180.0;
    let lonPhi = params.sunLongitude * Math.PI/180.0;
    let costheta = Math.cos(latTheta);
    let sintheta = Math.sin(latTheta);
    let cosphi = Math.cos(lonPhi);
    let sinphi = Math.sin(lonPhi);
    let x = sintheta * cosphi;
    let z = sintheta * sinphi;
    let y = costheta;
    params.sunDir = [x, y, z];
}

function fuzz_enabled()
{
    if (params.fuzz_weight == 0.0)
        return false;
    return true;
}

function coat_enabled()
{
    if (params.coat_weight == 0.0)
        return false;
    return true;
}

function volume_enabled()
{
    if (params.base_metalness == 1.0)
        return false;
    if (params.transmission_weight > 0.0 &&
        params.transmission_depth > 0.0)
        return true;
    if (params.subsurface_weight > 0.0)
        return true;
    return false;
}

function transmission_enabled()
{
    if (params.transmission_weight > 0.0)
        return true;
    if (params.subsurface_weight > 0.0)
        return true;
    return false;
}

function thin_film_enabled()
{
    if (params.thin_film_weight > 0.0)
        return true;
    return false;
}

function create_materials()
{
    renderer.outputColorSpace = LinearSRGBColorSpace;

    if (openpbrMaterial)
        openpbrMaterial.dispose();

    if (neutralMaterial)
        neutralMaterial.dispose();

    if (pathtracedMaterial)
        pathtracedMaterial.dispose();

    materialDefines.BOUNCES          = params.bounces;
    materialDefines.MAX_VOLUME_STEPS = params.max_volume_steps;
    materialDefines.FUZZ_ENABLED         = fuzz_enabled();
    materialDefines.COAT_ENABLED         = coat_enabled();
    materialDefines.TRANSMISSION_ENABLED = transmission_enabled();
    materialDefines.VOLUME_ENABLED       = volume_enabled();
    materialDefines.THIN_FILM_ENABLED    = thin_film_enabled();

    if (!PATHTRACING)
    {
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // openpbrMaterial (for rasterization)
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        openpbrMaterial = new ShaderMaterial( {

            defines: materialDefines,

            uniforms: UniformsUtils.merge( [

                    UniformsUtils.clone(ShaderLib.phong.uniforms),
                    {
                        cameraWorldMatrix:     { value: new Matrix4() },
                        invProjectionMatrix:   { value: new Matrix4() },
                        invModelMatrix:        { value: new Matrix4() },
                        resolution:            { value: new Vector2() },
                        samples:               { value: 0 },
                        accumulation_weight:   { value: 1 },

                        //////////////////////////////////////////////////////
                        // renderer
                        //////////////////////////////////////////////////////

                        wireframe:                           { value: params.wireframe, },
                        neutral_color:                       { value: new Vector3().fromArray(params.neutral_color) },
                        smooth_normals:                      { value: params.smooth_normals, },

                        //////////////////////////////////////////////////////
                        // lighting
                        //////////////////////////////////////////////////////

                        skyPower:                            { value: params.skyPower, },
                        skyColor:                            { value: array_to_vector3(params.skyColor) },

                        sunPower:                            { value: Math.pow(10.0,params.sunPower), },
                        sunAngularSize:                      { value: params.sunAngularSize, },
                        sunColor:                            { value: array_to_vector3(params.sunColor) },
                        sunDir:                              { value: array_to_vector3([0,0,0]) },

                        //////////////////////////////////////////////////////
                        // material
                        //////////////////////////////////////////////////////

                        base_weight:                         { value: params.base_weight },
                        base_color:                          { value: array_to_vector3(params.base_color) },
                        base_roughness:                      { value: params.base_roughness },
                        base_metalness:                      { value: params.base_metalness },

                        specular_weight:                     { value: params.specular_weight, },
                        specular_color:                      { value: array_to_vector3(params.specular_color) },
                        specular_roughness:                  { value: params.specular_roughness },
                        specular_anisotropy:                 { value: params.specular_anisotropy },
                        specular_rotation:                   { value: params.specular_rotation },
                        specular_ior:                        { value: params.specular_ior  },

                        transmission_weight:                 { value: params.transmission_weight, },
                        transmission_color:                  { value: array_to_vector3(params.transmission_color) },
                        transmission_depth:                  { value: params.transmission_depth },
                        transmission_scatter:                { value: array_to_vector3(params.transmission_scatter) },
                        transmission_scatter_anisotropy:     { value: params.transmission_scatter_anisotropy },
                        transmission_dispersion_abbe_number: { value: params.transmission_dispersion_abbe_number },
                        transmission_dispersion_scale:       { value: params.transmission_dispersion_scale },

                        subsurface_weight:                   { value: params.subsurface_weight },
                        subsurface_color:                    { value: array_to_vector3(params.subsurface_color) },
                        subsurface_radius:                   { value: params.subsurface_radius },
                        subsurface_radius_scale:             { value: array_to_vector3(params.subsurface_radius_scale) },
                        subsurface_anisotropy:               { value: params.subsurface_anisotropy },

                        coat_weight:                         { value: params.coat_weight },
                        coat_color:                          { value: array_to_vector3(params.coat_color) },
                        coat_roughness:                      { value: params.coat_roughness },
                        coat_anisotropy:                     { value: params.coat_anisotropy },
                        coat_rotation:                       { value: params.coat_rotation },
                        coat_ior:                            { value: params.coat_ior  },
                        coat_darkening:                      { value: params.coat_darkening  },

                        fuzz_weight:                         { value: params.fuzz_weight },
                        fuzz_color:                          { value: array_to_vector3(params.fuzz_color) },
                        fuzz_roughness:                      { value: params.fuzz_roughness },

                        emission_luminance:                  { value: params.emission_luminance },
                        emission_color:                      { value: array_to_vector3(params.emission_color) },

                        thin_film_weight:                    { value: params.thin_film_weight },
                        thin_film_thickness:                 { value: params.thin_film_thickness },
                        thin_film_ior:                       { value: params.thin_film_ior },

                        geometry_opacity:                    { value: params.geometry_opacity },
                        geometry_thin_walled:                { value: params.geometry_thin_walled }
                    }
                ] ),

                vertexShader:   glsl_rasterization_openpbr_vert,
                fragmentShader: glsl_rasterization_openpbr_frag,
                lights: true
            } );


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // neutralMaterial (for rasterization)
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        neutralMaterial = new ShaderMaterial( {

        defines: materialDefines,

        uniforms: UniformsUtils.merge( [

            UniformsUtils.clone(ShaderLib.phong.uniforms),
            {
                cameraWorldMatrix:     { value: new Matrix4() },
                invProjectionMatrix:   { value: new Matrix4() },
                invModelMatrix:        { value: new Matrix4() },
                resolution:            { value: new Vector2() },
                samples:               { value: 0 },
                accumulation_weight:   { value: 1 },

                //////////////////////////////////////////////////////
                // renderer
                //////////////////////////////////////////////////////

                wireframe:                           { value: params.wireframe, },
                smooth_normals:                      { value: params.smooth_normals, },

                //////////////////////////////////////////////////////
                // lighting
                //////////////////////////////////////////////////////

                skyPower:                            { value: params.skyPower, },
                skyColor:                            { value: array_to_vector3(params.skyColor) },
                sunPower:                            { value: Math.pow(10.0,params.sunPower), },
                sunAngularSize:                      { value: params.sunAngularSize, },
                sunColor:                            { value: array_to_vector3(params.sunColor) },
                sunDir:                              { value: array_to_vector3([0,0,0]) },

                //////////////////////////////////////////////////////
                // material
                //////////////////////////////////////////////////////

                neutral_color:                       { value: new Vector3().fromArray(params.neutral_color) }
            }
        ] ),

        vertexShader:   glsl_rasterization_neutral_vert,
        fragmentShader: glsl_rasterization_neutral_frag,
        lights: true
        } );
    }

    if (PATHTRACING)
    {
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // pathtracedMaterial (for pathtracing shader)
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        pathtracedMaterial = new ShaderMaterial( {

        defines: materialDefines,

        uniforms: UniformsUtils.merge( [

            UniformsUtils.clone(ShaderLib.phong.uniforms),
            {
                bvh_surface:             { value: new MeshBVHUniformStruct() },
                normalAttribute_surface: { value: new FloatVertexAttributeTexture() },
                tangentAttribute_surface:{ value: new FloatVertexAttributeTexture() },
                has_normals_surface:     { value: 1 },
                has_tangents_surface:    { value: 0 },

                bvh_props:             { value: new MeshBVHUniformStruct() },
                normalAttribute_props: { value: new FloatVertexAttributeTexture() },
                tangentAttribute_props:{ value: new FloatVertexAttributeTexture() },
                has_normals_props:     { value: 1 },
                has_tangents_props:    { value: 0 },

                cameraWorldMatrix:     { value: new Matrix4() },
                invProjectionMatrix:   { value: new Matrix4() },
                invModelMatrix:        { value: new Matrix4() },
                resolution:            { value: new Vector2() },

                samples:               { value: 0 },
                accumulation_weight:   { value: 1 },

                //////////////////////////////////////////////////////
                // renderer
                //////////////////////////////////////////////////////

                wireframe:                           { value: params.wireframe, },
                neutral_color:                       { value: new Vector3().fromArray(params.neutral_color) },
                smooth_normals:                      { value: params.smooth_normals, },

                //////////////////////////////////////////////////////
                // lighting
                //////////////////////////////////////////////////////

                skyPower:                            { value: params.skyPower, },
                skyColor:                            { value: array_to_vector3(params.skyColor) },

                sunPower:                            { value: Math.pow(10.0,params.sunPower), },
                sunAngularSize:                      { value: params.sunAngularSize, },
                sunColor:                            { value: array_to_vector3(params.sunColor) },
                sunDir:                              { value: array_to_vector3([0,0,0]) },

                //////////////////////////////////////////////////////
                // material
                //////////////////////////////////////////////////////

                base_weight:                         { value: params.base_weight },
                base_color:                          { value: array_to_vector3(params.base_color) },
                base_roughness:                      { value: params.base_roughness },
                base_metalness:                      { value: params.base_metalness },

                specular_weight:                     { value: params.specular_weight, },
                specular_color:                      { value: array_to_vector3(params.specular_color) },
                specular_roughness:                  { value: params.specular_roughness },
                specular_anisotropy:                 { value: params.specular_anisotropy },
                specular_rotation:                   { value: params.specular_rotation },
                specular_ior:                        { value: params.specular_ior  },

                transmission_weight:                 { value: params.transmission_weight, },
                transmission_color:                  { value: array_to_vector3(params.transmission_color) },
                transmission_depth:                  { value: params.transmission_depth },
                transmission_scatter:                { value: array_to_vector3(params.transmission_scatter) },
                transmission_scatter_anisotropy:     { value: params.transmission_scatter_anisotropy },
                transmission_dispersion_abbe_number: { value: params.transmission_dispersion_abbe_number },
                transmission_dispersion_scale:       { value: params.transmission_dispersion_scale },

                subsurface_weight:                   { value: params.subsurface_weight },
                subsurface_color:                    { value: array_to_vector3(params.subsurface_color) },
                subsurface_radius:                   { value: params.subsurface_radius },
                subsurface_radius_scale:             { value: array_to_vector3(params.subsurface_radius_scale) },
                subsurface_anisotropy:               { value: params.subsurface_anisotropy },

                coat_weight:                         { value: params.coat_weight },
                coat_color:                          { value: array_to_vector3(params.coat_color) },
                coat_roughness:                      { value: params.coat_roughness },
                coat_anisotropy:                     { value: params.coat_anisotropy },
                coat_rotation:                       { value: params.coat_rotation },
                coat_ior:                            { value: params.coat_ior  },
                coat_darkening:                      { value: params.coat_darkening  },

                fuzz_weight:                         { value: params.fuzz_weight },
                fuzz_color:                          { value: array_to_vector3(params.fuzz_color) },
                fuzz_roughness:                      { value: params.fuzz_roughness },

                emission_luminance:                  { value: params.emission_luminance },
                emission_color:                      { value: array_to_vector3(params.emission_color) },

                thin_film_weight:                    { value: params.thin_film_weight },
                thin_film_thickness:                 { value: params.thin_film_thickness },
                thin_film_ior:                       { value: params.thin_film_ior },

                geometry_opacity:                    { value: params.geometry_opacity },
                geometry_thin_walled:                { value: params.geometry_thin_walled }

            },
        ] ),

        vertexShader: `
            varying vec2 vUv;
            void main()
            {
                vec4 mvPosition = vec4( position, 1.0 );
                mvPosition = modelViewMatrix * mvPosition;
                gl_Position = projectionMatrix * mvPosition;
                vUv = uv;
            }
        `,

        fragmentShader: `precision highp isampler2D;
                            precision highp usampler2D;
                            precision highp int;
                            ${ shaderStructs }
                            ${ shaderIntersectFunction }
                        `
                        + glsl_pathtracing_main
                        + glsl_pathtracing_fuzz_brdf
                        + glsl_pathtracing_coat_brdf
                        + glsl_pathtracing_thin_film
                        + glsl_pathtracing_specular_brdf
                        + glsl_pathtracing_specular_btdf
                        + glsl_pathtracing_metal_brdf
                        + glsl_pathtracing_diffuse_brdf
                        + glsl_pathtracing_openpbr_surface
                        + glsl_pathtracing_pathtracer

        } );
    }
}

function init()
{
    // Setup progress bar spinner
    progress_bar = new Circle('#progress_overlay',
    {
        color: 'rgba(255, 128, 64, 0.75)',
        strokeWidth: 5.0,
        trailColor: 'rgba(255, 128, 64, 0.333)',
        trailWidth: 3.0,
        svgStyle: {
            display: 'block',
            width: '100%'
        },
        text: {
            value: '',
            className: 'progressbar__label',
            style: {
                color: 'rgba(169, 85, 42, 1.0)',
                position: 'absolute',
                fontWeight: 'bold',
                left: '50%',
                top: '50%',
                padding: 0,
                margin: 0,
                transform: {
                    prefix: true,
                    value: 'translate(-50%, -50%)'
                }
            },
            autoStyleContainer: true,
            alignToBottom: true
        },
        fill: null,
        duration: 2000.0,
        easing: 'linear',
        from: { color: 'rgba( 0,   0,  0, 0.0)' },
        to: {   color: 'rgba(32, 255, 32, 1.0)' },
        warnings: true
    });
    progress_bar.set(0.0);
    progress_bar.setText('');

    LOADED = false;
    MESH_SURFACE = null;
    MESH_PROPS = null;
    BVH_SURFACE = null;
    BVH_PROPS = null;

    // renderer setup
    renderer = new WebGLRenderer( { antialias: true, preserveDrawingBuffer: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setClearColor( 0x09141a );
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.outputColorSpace = LinearSRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMapSoft = true;
    renderer.shadowMap.type = PCFSoftShadowMap; // default THREE.PCFShadowMap
    renderer.physicallyBasedShading = true;

    document.body.appendChild( renderer.domElement );

    PATHTRACING = false;

    // stats setup
    stats = new Stats();
    document.body.appendChild( stats.dom );

    // Samples count text
    let samples_txt = document.getElementById('samples');
    samples_txt.style.visibility = 'visible';

    // Info text
    let info_txt = document.getElementById('info');
    info_txt.style.visibility = 'visible';

    gui = null;
    camera_initialized = false;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // initialize the scene and update the material properties with the bvh, materials, etc
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mesh_loader = new MeshLoader();

    load_scene(params.scene_name);
}

function load_geometry(scene_name)
{
    scene.background = env_map_texture;
    env_map_texture.mapping = EquirectangularReflectionMapping ;
    env_map_texture.colorSpace = SRGBColorSpace;
    if (!PATHTRACING)
    {
        neutralMaterial.envMap                = env_map_texture;
        neutralMaterial.uniforms.envMap.value = env_map_texture;
        openpbrMaterial.envMap                = env_map_texture;
        openpbrMaterial.uniforms.envMap.value = env_map_texture;
    }
    else
    {
        pathtracedMaterial.envMap                = env_map_texture;
        pathtracedMaterial.uniforms.envMap.value = env_map_texture;
    }

    // Load "neutral" objects (i.e. Lambert shaded background stuff)
    mesh_loader.load(scene_name + '/neutral_objects.glb').then( () => {

        if (!PATHTRACING)
        {
            // Set up mesh properties for rasterization
            mesh_loader.result.scene.traverse((o) => {
                if (o.isMesh)
                {
                    o.material = neutralMaterial;
                    o.receiveShadow = true;
                    o.castShadow = false;
                    o.material.side = DoubleSide;
                }
            });
        }

        scene.add(mesh_loader.result.scene);

        MESH_PROPS = mesh_loader.result.mesh;

        if (PATHTRACING)
        {
            // Set up mesh properties for pathtracing
            BVH_PROPS  = mesh_loader.result.bvh;
            pathtracedMaterial.uniforms.bvh_props.value.updateFrom( BVH_PROPS );
            pathtracedMaterial.uniforms.has_normals_props.value = false;
            pathtracedMaterial.uniforms.has_tangents_props.value = false;
            if (MESH_PROPS.geometry.attributes.normal)
            {
                pathtracedMaterial.uniforms.normalAttribute_props.value.updateFrom( MESH_PROPS.geometry.attributes.normal );
                pathtracedMaterial.uniforms.has_normals_props.value = true;
            }
            if (MESH_PROPS.geometry.attributes.tangent)
            {
                pathtracedMaterial.uniforms.tangentAttribute_props.value.updateFrom( MESH_PROPS.geometry.attributes.tangent );
                pathtracedMaterial.uniforms.has_tangents_props.value = true;
            }
            console.log("  has_normals_scene:  ", pathtracedMaterial.uniforms.has_normals_props);
            console.log("  has_tangents_scene: ", pathtracedMaterial.uniforms.has_tangents_props);
        }

        progress_bar.animate(0.5);
        mesh_loader.reset();

        // Load OpenPBR-shaded objects
        mesh_loader.load(scene_name + '/openpbr_objects.glb').then( () => {

            if (!PATHTRACING)
            {
                // Set up mesh properties for rasterization
                mesh_loader.result.scene.traverse((o) => {
                    if (o.isMesh)
                    {
                        o.material = openpbrMaterial;
                        o.receiveShadow = true;
                        o.castShadow = true;
                    }
                });
            }

            scene.add(mesh_loader.result.scene);

            MESH_SURFACE = mesh_loader.result.mesh;

            if (PATHTRACING)
            {
                // Set up mesh properties for pathtracing
                BVH_SURFACE  = mesh_loader.result.bvh;
                pathtracedMaterial.uniforms.bvh_surface.value.updateFrom( BVH_SURFACE );
                pathtracedMaterial.uniforms.has_normals_surface.value = false;
                pathtracedMaterial.uniforms.has_tangents_surface.value = false;
                if (MESH_SURFACE.geometry.attributes.normal)
                {
                    pathtracedMaterial.uniforms.normalAttribute_surface.value.updateFrom( MESH_SURFACE.geometry.attributes.normal );
                    pathtracedMaterial.uniforms.has_normals_surface.value = true;
                }
                if (MESH_SURFACE.geometry.attributes.tangent)
                {
                    pathtracedMaterial.uniforms.tangentAttribute_surface.value.updateFrom( MESH_SURFACE.geometry.attributes.tangent );
                    pathtracedMaterial.uniforms.has_tangents_surface.value = true;
                }
                console.log("  has_normals_surface:  ", pathtracedMaterial.uniforms.has_normals_surface);
                console.log("  has_tangents_surface: ", pathtracedMaterial.uniforms.has_tangents_surface);
                console.log("===> LOADED");
            }

            LOADED = true;

            post_load_setup();

            progress_bar.animate(1.0);
            let progress_overlay = document.getElementById('progress_overlay');
            progress_finished_timer = performance.now();

        } )

    } );
}

function load_scene(scene_name)
{
    console.log('Loading scene: ', scene_name);
    LOADED = false;

    create_materials()

    ////////////////////////////////////////////////////////////////////////////////////
    // Create three.js scene
    scene = new Scene();
    ////////////////////////////////////////////////////////////////////////////////////

    progress_bar.setText('loading meshes...');
    progress_bar.animate(0.0);

    // Load env map
    if (!env_map_texture)
    {
        const textureLoader = new TextureLoader();
        let env_map_path = 'textures/envmaps/etzwihl_16k.jpg';
        let textureEquirec = textureLoader.load(env_map_path,
            function (texture) {
                console.log('-> loaded env map: ', env_map_path);
                    env_map_texture = texture;
                    load_geometry(scene_name);
                }
        );
    }
    else
        load_geometry(scene_name);
}

function reset_camera(scene_name)
{
    let camera_fov = 23.6701655;
    let camera_near = 0.01;
    let camera_far = 1000.0;
    camera = new PerspectiveCamera( camera_fov, window.innerWidth / window.innerHeight, camera_near, camera_far );

    orbitControls = new OrbitControls( camera, renderer.domElement );
    orbitControls.addEventListener( 'change', () => { resetSamples(); } );
    let matrixWorld = new Matrix4();

    if (scene_name == 'standard-shader-ball')
    {
        // Set camera default orientation according to the Standard Shader Ball USD asset description:
        matrixWorld.set( 0.9396926207859084,                  0, -0.3420201433256687, 0,
                        -0.2203032561704394, 0.7649214009184319, -0.6052782217606094, 0,
                        0.26161852717499334, 0.6441236297613865,  0.7187909959242699, 0,
                        6.531538924716362,               19.5,  17.948521838355774, 1 );
    }
    else if (scene_name == 'glavenus')
    {
        matrixWorld.set( 0.4848291963218869, -6.938893903907228e-18, -0.8746088556571293,   0,
                        -0.07533009256065425, 0.9962839037303908,    -0.041758356319859954, 0,
                            0.8713587249512548,  0.08613003638530015,    0.4830275243540376,   0,
                        23.076273094000275,   6.7653774216248,        14.822630983786677,   1);

    }
    else if (scene_name == 'terrain')
    {
        matrixWorld.set( 0.7242953632536803, -1.1102230246251565e-16, -0.6894898307946385, 0,
                        -0.4511571209928634,  0.7562050657737049,     -0.4739315886028461, 0,
                            0.5213957028463604,  0.6543346991396579,      0.5477158228088388, 0,
                            8.561709328489492,  11.460860759783042,       8.95672568146927,   1);
    }
    else if (scene_name == 'bearded-man')
    {
        matrixWorld.set(0.6586894440882616, -1.3877787807814457e-17, 0.752414922929295,   0,
                        0.13367205033823076, 0.9840924050751759,    -0.11702102901499911, 0,
                        -0.7404458111199431,  0.17765736200156684,    0.648211279230448,   0,
                        -20.089277049402824,   9.131027464916848,     18.02162149148976,    1);
    }

    matrixWorld.transpose();
    camera.matrixAutoUpdate = false;
    camera.applyMatrix4(matrixWorld);
    camera.matrixAutoUpdate = true;
    camera.updateMatrixWorld();

    let dir = new Vector3();
    camera.getWorldDirection(dir);
    let cam_target = camera.position.clone();
    cam_target.addScaledVector(dir, 23.39613);
    orbitControls.target.copy(cam_target);

    orbitControls.zoomSpeed = 1.5;
    orbitControls.flySpeed = 0.01;
    orbitControls.update();
}


function setup_gui()
{
    if (gui)
        gui.destroy()
    gui = new GUI({ width: 300 });

    ///// Material folder /////////////////////////////////////
    const material_folder = gui.addFolder('Material');

    // Base folder
    const base_folder = material_folder.addFolder('Base');
    base_folder.add(params,          'base_weight', 0.0, 1.0).onChange(                               v => { resetSamples(); });
    base_folder.addColor(params,     'base_color').onChange(                                          v => { resetSamples(); });
    base_folder.add(params,          'base_roughness', 0.0, 1.0).onChange(                            v => { resetSamples(); });
    base_folder.add(params,          'base_metalness', 0.0, 1.0).onChange(                            v => { resetSamples(); });

    // Specular folder
    const specular_folder = material_folder.addFolder('Specular');
    specular_folder.add(params,      'specular_weight', 0.0, 1.0).onChange(                           v => { resetSamples(); });
    specular_folder.addColor(params, 'specular_color').onChange(                                      v => { resetSamples(); });
    specular_folder.add(params,      'specular_roughness', 0.0, 1.0).onChange(                        v => { resetSamples(); });
    specular_folder.add(params,      'specular_ior', 1.0, 5.0).onChange(                              v => { resetSamples(); });
    specular_folder.add(params,      'specular_anisotropy', 0.0, 1.0).onChange(                       v => { resetSamples(); });
    specular_folder.add(params,      'specular_rotation', 0.0, 1.0).onChange(                         v => { resetSamples(); });

    // Transmission folder
    const transmission_folder = material_folder.addFolder('Transmission');
    transmission_folder.add(params,      'transmission_weight', 0.0, 1.0).onChange(                   v => { resetSamples(); });
    transmission_folder.addColor(params, 'transmission_color').onChange(                              v => { resetSamples(); });
    transmission_folder.add(params,      'transmission_depth', 0.0, 1.0).onChange(                    v => { resetSamples(); });
    transmission_folder.addColor(params, 'transmission_scatter').onChange(                            v => { resetSamples(); });
    transmission_folder.add(params,      'transmission_scatter_anisotropy', -1.0, 1.0).onChange(      v => { resetSamples(); });
    transmission_folder.add(params,      'transmission_dispersion_abbe_number', 9.0, 91.0).onChange(  v => { resetSamples(); });
    transmission_folder.add(params,      'transmission_dispersion_scale', 0.0, 1.0).onChange(         v => { resetSamples(); });
    transmission_folder.close();

    // Subsurface folder
    const subsurface_folder = material_folder.addFolder('Subsurface');
    subsurface_folder.add(params,      'subsurface_weight', 0.0, 1.0).onChange(                       v => { resetSamples(); });
    subsurface_folder.addColor(params, 'subsurface_color').onChange(                                  v => { resetSamples(); });
    subsurface_folder.add(params,      'subsurface_radius', 0.0, 1.0).onChange(                       v => { resetSamples(); });
    subsurface_folder.addColor(params, 'subsurface_radius_scale').onChange(                           v => { resetSamples(); });
    subsurface_folder.add(params,      'subsurface_anisotropy', -1.0, 1.0).onChange(                  v => { resetSamples(); });
    subsurface_folder.close();

    // Coat folder
    const coat_folder = material_folder.addFolder('Coat');
    coat_folder.add(params,          'coat_weight', 0.0, 1.0).onChange(                               v => { resetSamples(); });
    coat_folder.addColor(params,     'coat_color').onChange(                                          v => { resetSamples(); });
    coat_folder.add(params,          'coat_roughness', 0.0, 1.0).onChange(                            v => { resetSamples(); });
    coat_folder.add(params,          'coat_ior', 1.0, 3.0).onChange(                                  v => { resetSamples(); });
    coat_folder.add(params,          'coat_anisotropy', 0.0, 1.0).onChange(                           v => { resetSamples(); });
    coat_folder.add(params,          'coat_rotation', 0.0, 1.0).onChange(                             v => { resetSamples(); });
    coat_folder.add(params,          'coat_darkening', 0.0, 1.0).onChange(                            v => { resetSamples(); });
    coat_folder.close();

    // Fuzz folder
    const fuzz_folder = material_folder.addFolder('Fuzz');
    fuzz_folder.add(params,          'fuzz_weight', 0.0, 1.0).onChange(                               v => { resetSamples(); });
    fuzz_folder.addColor(params,     'fuzz_color').onChange(                                          v => { resetSamples(); });
    fuzz_folder.add(params,          'fuzz_roughness', 0.0, 1.0).onChange(                            v => { resetSamples(); });
    fuzz_folder.close();

    // Emission folder
    const emission_folder = material_folder.addFolder('Emission');
    emission_folder.add(params,          'emission_luminance', 0.0, 10.0).onChange(                   v => { resetSamples(); });
    emission_folder.addColor(params,     'emission_color').onChange(                                  v => { resetSamples(); });
    emission_folder.close();

    // Thin-film folder
    const thin_film_folder = material_folder.addFolder('Thin Film');
    thin_film_folder.add(params,          'thin_film_weight', 0.0, 1.0).onChange(                     v => { resetSamples(); });
    thin_film_folder.add(params,          'thin_film_thickness', 0.0, 2000.0).onChange(               v => { resetSamples(); });
    thin_film_folder.add(params,          'thin_film_ior', 1.0, 3.0).onChange(                        v => { resetSamples(); });
    thin_film_folder.close();

    // geometry folder
    const geometry_folder = material_folder.addFolder('Geometry');
    geometry_folder.add(params,      'geometry_opacity', 0.0, 1.0).onChange(                          v => { resetSamples(); });
    geometry_folder.add(params,      'geometry_thin_walled').onChange(                                v => { resetSamples(); });
    geometry_folder.close();

    ///// Lighting folder /////////////////////////////////////
    const lighting_folder = gui.addFolder('Lighting');
    lighting_folder.add(params, 'skyPower', 0.0, 2.0).onChange(                                       v => { resetSamples(); });
    lighting_folder.addColor(params, 'skyColor').onChange(                                            v => { resetSamples(); });
    lighting_folder.add(params, 'sunPower', -4.0, 4.0).onChange(                                      v => { resetSamples(); });
    lighting_folder.add(params, 'sunAngularSize', 0.0, 40.0).onChange(                                v => { resetSamples(); });
    lighting_folder.add(params, 'sunLatitude', 0.0, 90.0).onChange(                                   v => { resetSamples(); });
    lighting_folder.add(params, 'sunLongitude', 0.0, 360.0).onChange(                                 v => { resetSamples(); });
    lighting_folder.addColor(params, 'sunColor').onChange(                                            v => { resetSamples(); });
    lighting_folder.close();

    ///// Renderer folder /////////////////////////////////////
    const renderer_folder = gui.addFolder('Renderer');
    renderer_folder.add(params, 'scene_name', scene_names).onChange(                                  v => { load_scene(v); });
    renderer_folder.add( params, 'smooth_normals' ).onChange(                                         v => { resetSamples(); });
    renderer_folder.add( params, 'wireframe' ).onChange(                                              v => { resetSamples(); });
    renderer_folder.addColor(params, 'neutral_color').onChange(                                       v => { resetSamples(); });
    renderer_folder.add( params, 'bounces', 0, 100, 1 ).onChange(                                     v => { materialDefines.BOUNCES = parseInt( v );
                                                                                                             load_scene(params.scene_name); } );
    renderer_folder.add( params, 'max_samples' ).onChange(                                            v => { load_scene(params.scene_name); });
    renderer_folder.add( params, 'max_volume_steps', 1, 100, 1 ).onChange(                            v => { materialDefines.MAX_VOLUME_STEPS = parseInt( v );
                                                                                                             load_scene(params.scene_name); } );
    renderer_folder.close();

    gui.add( params, 'reset_camera' );
    gui.open();
}

function post_load_setup()
{
    console.log(scene);

    if (!PATHTRACING)
    {
        //////////////////////////////////////////////////////////
        // Setup THREE.js lighting
        //////////////////////////////////////////////////////////

        directionalLight = new DirectionalLight(0xffffff, 1.0);

        updateSunDir()
        let dL = 20.0;
        directionalLight.position.set(MESH_PROPS.position[0] + dL*params.sunDir[0],
                                      MESH_PROPS.position[1] + dL*params.sunDir[1],
                                      MESH_PROPS.position[2] + dL*params.sunDir[2]);
        directionalLight.target.position.copy( MESH_PROPS.position );
        directionalLight.castShadow = true; // default false

        //Set up shadow properties for the directionalLight
        directionalLight.shadow.mapSize.width  = 2048; // default
        directionalLight.shadow.mapSize.height = 2048; // default

        let shadow_extent = 9.0;
        directionalLight.shadow.camera.left   = -shadow_extent;
        directionalLight.shadow.camera.right  =  shadow_extent;
        directionalLight.shadow.camera.bottom = -shadow_extent;
        directionalLight.shadow.camera.top    =  shadow_extent;
        directionalLight.shadow.camera.near = 10.0; // default
        directionalLight.shadow.camera.far = 100.0; // default
        directionalLight.shadowDarkness = 0.0;
        scene.add( directionalLight );

        ambientLight = new AmbientLight( 0x0 );
        scene.add( ambientLight );

        // NB, add this helper to debug shadow map issues
        //const helper = new CameraHelper(directionalLight.shadow.camera);
        //scene.add(helper);
    }

    if (PATHTRACING)
    {
        //////////////////////////////////////////////////////////
        // Setup framebuffers for pathtracing
        //////////////////////////////////////////////////////////
        pathtracedQuad = new FullScreenQuad( pathtracedMaterial );

        pathtracedMaterial.transparent = true;
        pathtracedMaterial.depthWrite = false;

        pathtracingRenderTarget = new WebGLRenderTarget(1, 1, {format: RGBAFormat, type: FloatType, colorSpace: LinearSRGBColorSpace});
        pathtracedFinalQuad = new FullScreenQuad( new MeshBasicMaterial({map: pathtracingRenderTarget.texture}) );
    }

    // Trigger initial shader compile
    trigger_recompile();

    //////////////////////////////////////////////////////////
    // Setup camera
    //////////////////////////////////////////////////////////
    if (!camera_initialized)
        reset_camera(params.scene_name);
    camera_initialized = true;

    //////////////////////////////////////////////////////////
    // Setup GUI
    //////////////////////////////////////////////////////////
    setup_gui();

    //////////////////////////////////////////////////////////
    // Setup window
    //////////////////////////////////////////////////////////
    window.addEventListener( 'resize', resize, false );
    resize();
}

function trigger_recompile()
{
    let tmp_cam = new OrthographicCamera( - 1, 1, 1, - 1, 0, 1 );
    startCompilationProgress();
    let compile_promise = renderer.compileAsync(scene, tmp_cam);
    compile_promise.then((val) => {
        console.log('shaders successfully compiled.');
        finishCompilationProgress();
    }).catch((err) => {
        console.log('shader compilation error: ' + err);
    }).finally(() => {});
    }

function startCompilationProgress()
{
    console.log('startCompilationProgress');
    let progress_overlay = document.getElementById('progress_overlay');
    progress_overlay.style.display = 'block';
    progress_overlay.style.opacity = 1;
    progress_bar.set(0.0);
    progress_bar.setText('shaders compiling...');
    COMPILING = true;
}

function finishCompilationProgress()
{
    console.log('finishCompilationProgress');
    progress_bar.set(1.0);
    progress_finished_timer = performance.now();
    COMPILING = false;
}

function resize()
{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize( w, h );
    renderer.setPixelRatio(1.0);
    if (PATHTRACING)
        pathtracingRenderTarget.setSize(w, h);
    resetSamples();
}

function get_vector3(array3)
{
    return new Vector3(array3[0], array3[1], array3[2]);
}

function resetSamples()
{
    samples = 0;

    if (PATHTRACING)
    {
        let reload = false;
        reload = reload || (materialDefines.FUZZ_ENABLED         != fuzz_enabled());
        reload = reload || (materialDefines.COAT_ENABLED         != coat_enabled());
        reload = reload || (materialDefines.TRANSMISSION_ENABLED != transmission_enabled());
        reload = reload || (materialDefines.VOLUME_ENABLED       != volume_enabled());
        reload = reload || (materialDefines.THIN_FILM_ENABLED    != thin_film_enabled());
        if (reload)
            load_scene(params.scene_name);
    }
}

function fadeOutProgressBar(time_ms)
{
    let progress_overlay = document.getElementById('progress_overlay');
    var fadeOutEffect = setInterval(function () {
    if (!progress_overlay.style.opacity) {
        progress_overlay.style.opacity = 1;
    }
    if (progress_overlay.style.opacity > 0) {
        progress_overlay.style.opacity -= 0.025;
    } else {
        progress_overlay.style.display = 'none';
        progress_overlay.style.opacity = 0;
        clearInterval(fadeOutEffect);
    }
    }, time_ms);
}

function sync_shader_uniforms(uniforms)
{
    const w = window.innerWidth;
    const h = window.innerHeight;

    // sync camera
    uniforms.cameraWorldMatrix.value.copy( camera.matrixWorld );
    uniforms.invProjectionMatrix.value.copy( camera.projectionMatrixInverse );
    uniforms.invModelMatrix.value.copy( scene.matrixWorld ).invert();

    // sync renderer params
    let resolution = new Vector2(w, h);
    uniforms.resolution.value.copy(resolution);
    uniforms.accumulation_weight.value                    = 1.0 / (samples + 1.0); // implements Monte-Carlo accumulation
    uniforms.samples.value                                = samples;

    uniforms.wireframe.value                              = params.wireframe;
    uniforms.neutral_color.value.copy(get_vector3(          params.neutral_color));
    uniforms.smooth_normals.value                         = params.smooth_normals;

    // sync material params
    uniforms.base_weight.value                            = params.base_weight;
    uniforms.base_color.value.copy(get_vector3(             params.base_color));
    uniforms.base_roughness.value                         = params.base_roughness;
    uniforms.base_metalness.value                         = params.base_metalness;

    uniforms.specular_weight.value                        = params.specular_weight;
    uniforms.specular_color.value.copy(get_vector3(         params.specular_color));
    uniforms.specular_roughness.value                     = params.specular_roughness;
    uniforms.specular_anisotropy.value                    = params.specular_anisotropy;
    uniforms.specular_rotation.value                      = params.specular_rotation;
    uniforms.specular_ior.value                           = params.specular_ior;

    uniforms.transmission_weight.value                    = params.transmission_weight;
    uniforms.transmission_color.value.copy(get_vector3(     params.transmission_color));
    uniforms.transmission_depth.value                     = params.transmission_depth;
    uniforms.transmission_scatter.value.copy(get_vector3(   params.transmission_scatter));
    uniforms.transmission_scatter_anisotropy.value        = params.transmission_scatter_anisotropy;
    uniforms.transmission_dispersion_abbe_number.value    = params.transmission_dispersion_abbe_number;
    uniforms.transmission_dispersion_scale.value          = params.transmission_dispersion_scale;

    uniforms.subsurface_weight.value                      = params.subsurface_weight;
    uniforms.subsurface_color.value.copy(get_vector3(       params.subsurface_color));
    uniforms.subsurface_radius.value                      = params.subsurface_radius;
    uniforms.subsurface_radius_scale.value.copy(get_vector3(params.subsurface_radius_scale));
    uniforms.subsurface_anisotropy.value                  = params.subsurface_anisotropy;

    uniforms.coat_weight.value                            = params.coat_weight;
    uniforms.coat_color.value.copy(get_vector3(             params.coat_color));
    uniforms.coat_roughness.value                         = params.coat_roughness;
    uniforms.coat_anisotropy.value                        = params.coat_anisotropy;
    uniforms.coat_rotation.value                          = params.coat_rotation;
    uniforms.coat_ior.value                               = params.coat_ior;
    uniforms.coat_darkening .value                        = params.coat_darkening;

    uniforms.fuzz_weight.value                            = params.fuzz_weight;
    uniforms.fuzz_color.value.copy(get_vector3(             params.fuzz_color));
    uniforms.fuzz_roughness.value                         = params.fuzz_roughness;

    uniforms.emission_luminance.value                     = params.emission_luminance;
    uniforms.emission_color.value.copy(get_vector3(         params.emission_color));

    uniforms.thin_film_weight.value                       = params.thin_film_weight;
    uniforms.thin_film_thickness.value                    = params.thin_film_thickness;
    uniforms.thin_film_ior.value                          = params.thin_film_ior;

    uniforms.geometry_opacity.value                       = params.geometry_opacity;
    uniforms.geometry_thin_walled.value                   = params.geometry_thin_walled;

    // sync lighting params
    uniforms.skyPower.value                               = params.skyPower;
    uniforms.skyColor.value.copy(get_vector3(               params.skyColor));

    uniforms.sunPower.value                               = Math.pow(10.0, params.sunPower);
    uniforms.sunAngularSize.value                         = params.sunAngularSize;
    uniforms.sunColor.value.copy(get_vector3(               params.sunColor));
    updateSunDir();
    uniforms.sunDir.value.copy(get_vector3(                 params.sunDir));
}

function render()
{
    console.log('render()')
    if (!LOADED)
    {
        console.log('not LOADED')
        requestAnimationFrame( render );
        return;
    }

    renderer.domElement.style.imageRendering = 'auto';

    if (samples >= params.max_samples)
    {
        requestAnimationFrame( render );
        return;
    }

    if (!COMPILING && LOADED)
    {
        camera.updateMatrixWorld();

        //////////////////////////////////////////////////////
        // render framebuffer
        //////////////////////////////////////////////////////

        if (PATHTRACING)
        {
            sync_shader_uniforms(pathtracedMaterial.uniforms);

            // render float target
            renderer.autoClear = (samples === 0);
            renderer.setRenderTarget( pathtracingRenderTarget );
            pathtracedQuad.render( renderer );

            // render to screen
            renderer.setRenderTarget( null );
            renderer.autoClear = true;
            pathtracedFinalQuad.render( renderer );

            samples++;
        }
        else
        {

            renderer.setRenderTarget( null );
            sync_shader_uniforms(openpbrMaterial.uniforms);
            neutralMaterial.uniforms.neutral_color.value.copy(get_vector3(params.neutral_color));
            neutralMaterial.uniforms.skyPower.value                               = params.skyPower;
            neutralMaterial.uniforms.skyColor.value.copy(get_vector3(               params.skyColor));
            neutralMaterial.uniforms.sunPower.value                               = Math.pow(10.0, params.sunPower);
            neutralMaterial.uniforms.sunColor.value.copy(get_vector3(               params.sunColor));
            neutralMaterial.uniforms.sunDir.value.copy(get_vector3(                 params.sunDir));

            let dL = 20.0;
            directionalLight.position.set(MESH_PROPS.position.x + dL*params.sunDir[0],
                                          MESH_PROPS.position.y + dL*params.sunDir[1],
                                          MESH_PROPS.position.z + dL*params.sunDir[2]);
            directionalLight.intensity = Math.pow(10.0, params.sunPower);
            let sunColor3 = new Color(params.sunColor[0], params.sunColor[1], params.sunColor[2]);
            directionalLight.color = sunColor3.getHex();
            directionalLight.updateMatrix();

            let skyColor3 = new Color(params.skyColor[0], params.skyColor[1], params.skyColor[2]);
            ambientLight.color = skyColor3.getHex();
            ambientLight.intensity = params.skyPower;

            renderer.shadowMap.needsUpdate = true;

            camera.updateProjectionMatrix();
            camera.clearViewOffset();

            renderer.setRenderTarget( null );
            renderer.autoClear = true;
            renderer.render( scene, camera );
        }
    }
    else
    {
        resetSamples();
        camera.updateProjectionMatrix();
        camera.clearViewOffset();
        renderer.render( scene, camera );
        renderer.autoClear = true;
    }

    // Text HUD update
    let samples_txt = document.getElementById('samples');
    let    info_txt = document.getElementById('info');
    if (PATHTRACING)
    {
        samples_txt.style.visibility = 'visible';
        samples_txt.innerText = `samples: ${ samples }`;
        info_txt.innerText = `OpenPBR viewer, pathtracing mode (press 'R' for rasterization)`;
    }
    else
    {
        samples_txt.style.visibility = 'hidden';
        info_txt.innerText = `OpenPBR viewer, rasterization mode (press 'R' for pathtracing)`;
    }

    // Progress spinner update
    if (!COMPILING)
    {
        let progress_overlay = document.getElementById('progress_overlay');
        let progress_bar_visible = progress_overlay.style.display != 'none';
        if (progress_bar_visible)
        {
            let time_since_progress_finished_ms = performance.now() - progress_finished_timer;
            if (time_since_progress_finished_ms > 300.0)
                fadeOutProgressBar(300);
        }
    }
    if (COMPILING)
    {
        console.log('COMPILING...');
        if (progress_bar.value() < 0.01)
            progress_bar.animate(1.0);
        else if (progress_bar.value() > 0.99)
        {
            progress_bar.set(0.0);
            progress_bar.animate(1.0);
        }
    }

    stats.update();

    requestAnimationFrame( render );
}


document.onkeydown = function (event)
{
    event = event || window.event;
    var charCode = (event.which) ? event.which : event.keyCode;
    switch (charCode)
    {
        case 122: // F11 key: go fullscreen
        {
            var element	= document.body;
            if      ( 'webkitCancelFullScreen' in document ) element.webkitRequestFullScreen();
            else if ( 'mozCancelFullScreen'    in document ) element.mozRequestFullScreen();
            else console.assert(false);
            orbitControls.update();
            resetSamples();
            break;
        }
        case 70: // F key: reset cam
        {
            reset_camera(params.scene_name);
            break;
        }
        case 72: // H key: toggle hide/show gui
        {
            gui.show( gui._hidden );
            if (document.body.contains(stats.dom)) document.body.removeChild( stats.dom );
            else                                   document.body.appendChild( stats.dom );
            let info_txt = document.getElementById('info');
            if (info_txt.style.visibility == 'visible') info_txt.style.visibility = 'hidden';
            else                                        info_txt.style.visibility = 'visible';
            let samples_txt = document.getElementById('samples');
            if (samples_txt.style.visibility == 'visible') samples_txt.style.visibility = 'hidden';
            else                                           samples_txt.style.visibility = 'visible';
            break;
        }
        case 87: // W key: cam forward
        {
            let toTarget = new Vector3();
            toTarget.copy(orbitControls.target);
            toTarget.sub(camera.position);
            let distToTarget = toTarget.length();
            toTarget.normalize();
            var move = new Vector3();
            move.copy(toTarget);
            move.multiplyScalar(orbitControls.flySpeed*distToTarget);
            camera.position.add(move);
            orbitControls.target.add(move);
            orbitControls.update();
            resetSamples();
            break;
        }
        case 65: // A key: cam left
        {
            let toTarget = new Vector3();
            toTarget.copy(orbitControls.target);
            toTarget.sub(camera.position);
            let distToTarget = toTarget.length();
            var localX = new Vector3(1.0, 0.0, 0.0);
            var worldX = localX.transformDirection( camera.matrix );
            var move = new Vector3();
            move.copy(worldX);
            move.multiplyScalar(-orbitControls.flySpeed*distToTarget);
            camera.position.add(move);
            orbitControls.target.add(move);
            orbitControls.update();
            resetSamples();
            break;
        }
        case 83: // S key: cam back
        {
            let toTarget = new Vector3();
            toTarget.copy(orbitControls.target);
            toTarget.sub(camera.position);
            let distToTarget = toTarget.length();
            toTarget.normalize();
            var move = new Vector3();
            move.copy(toTarget);
            move.multiplyScalar(-orbitControls.flySpeed*distToTarget);
            camera.position.add(move);
            orbitControls.target.add(move);
            orbitControls.update();
            resetSamples();
            break;
        }
        case 68: // D key: cam right
        {
            let toTarget = new Vector3();
            toTarget.copy(orbitControls.target);
            toTarget.sub(camera.position);
            let distToTarget = toTarget.length();
            var localX = new Vector3(1.0, 0.0, 0.0);
            var worldX = localX.transformDirection( camera.matrix );
            var move = new Vector3();
            move.copy(worldX);
            move.multiplyScalar(orbitControls.flySpeed*distToTarget);
            camera.position.add(move);
            orbitControls.target.add(move);
            orbitControls.update();
            resetSamples();
            break;
        }
        case 80: // P key: save current image to disk
        {
            var link = document.createElement('a');
            let filename = `openpbr-viewer-screenshot.png`;
            link.download = filename;
            renderer.domElement.toBlob(function(blob){
                    link.href = URL.createObjectURL(blob);
                    var event = new MouseEvent('click');
                    link.dispatchEvent(event);
                    requestAnimationFrame( render );
                },'image/png', 1);
            break;
        }

        case 82: // R key: toggle between rasterizer and pathtracer
        {
            PATHTRACING = !PATHTRACING;
            load_scene(params.scene_name);
            break;
        }
    }
}