//
//  neuron_segmentation.cpp
//  cgal_testing
//
//  Created by Brendan Celii on 11/25/18.
//  Copyright © 2018 Brendan Celii. All rights reserved.
//

#include "neuron_cgal_segmentation.hpp"

//
//  segmentation_from_sdf_value_example_2.cpp
//  cgal_testing
//
//  Created by Brendan Celii on 11/1/18.
//  Copyright © 2018 Brendan Celii. All rights reserved.
//
#include <Python.h>
//#include "segmentation_from_sdf_value_example_2.hpp"

#if 0
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <CGAL/IO/OFF_reader.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <vector>
#endif

#if 1
#include <CGAL/IO/OFF_reader.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;


//PyObject *cgal_segmentation(PyObject *self, PyObject *args)
//const char * cgal_segmentation(const char *location,const char * filename,int number_of_clusters,double smoothing_lambda)
int cgal_segmentation(const char* location_with_filename, int number_of_clusters,double smoothing_lambda)
{
    
    // create and read Polyhedron
    Polyhedron mesh;
    /*std::ifstream input2("helloWorld.txt");
     if ( !input2){
     std::cerr << "!input for helloWorld" << std::endl;
     return EXIT_FAILURE;
     }*/
    
    
    /*if ( !input){
     std::cerr << "!input." << std::endl;
     return EXIT_FAILURE;
     }
     else{
     std::cout << "!input. false" << std::endl;
     }
     
     
     if ( !(input >> mesh) ){
     std::cerr << "!(input >> mesh)" << std::endl;
     return EXIT_FAILURE;
     }
     else{
     std::cout << "!(input >> mesh) false" << std::endl;
     }
     if ( mesh.empty() ){
     std::cerr << "mesh.empty()" << std::endl;
     return EXIT_FAILURE;
     }
     else{
     std::cout << "mesh.empty() false" << std::endl;
     }
     
     if ( ( !CGAL::is_triangle_mesh(mesh))) {
     std::cerr << "( !CGAL::is_triangle_mesh(mesh))" << std::endl;
     return EXIT_FAILURE;
     }
     else{
     std::cout << "( !CGAL::is_triangle_mesh(mesh)) false" << std::endl;
     }
     std::cout << (!input ) << std::endl;
     std::cout << ( !(input >> mesh)) << std::endl;
     std::cout << (  mesh.empty()) << std::endl;
     std::cout << (  ( !CGAL::is_triangle_mesh(mesh))) << std::endl;
     std::cout << (!input || !(input >> mesh)) << std::endl;*/
    
    //HAD TO MANUALLY SET THE WORKING DIRECTORY TO GET THE FILE TO BE RECOGNIZED
    //If you have a line of code that does input >> mesh anywhere before, it messes up the whole process
    /*char final[1000];
     strcpy(final,location);
     strcat(final,filename);
     const char ext[8] = ".off";
     strcat(final,ext);
     std::ifstream input(final);*/
    //std::string location = "/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/auto_segmented_big_segments/";
    //std::string filename = "neuron_28571618_Basal_1";
    //std::string filename_str(filename);
    //std::string location_str(location);
    std::string location_with_filename_str(location_with_filename);
    std::string input_file_name = location_with_filename_str + ".off";
    std::ifstream input(input_file_name.c_str());
    
    
    /* New way of importing the mesh */
    if (!input)
    {
        std::cerr << "Cannot open file " << std::endl;
        return 2;
    }
    std::vector<K::Point_3> points;
    std::vector< std::vector<std::size_t> > polygons;
    if (!CGAL::read_OFF(input, points, polygons))
    {
        std::cerr << "Error parsing the OFF file " << std::endl;
        return 3;
    }
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if (CGAL::is_closed(mesh) && (!CGAL::Polygon_mesh_processing::is_outward_oriented(mesh)))
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);
    
    if(mesh.empty()){
        return 4;
    }
    if(( !CGAL::is_triangle_mesh(mesh))){
        return 6;
    }
    
    
    /*
     if ( !input || !(input >> mesh) || mesh.empty()  || ( !CGAL::is_triangle_mesh(mesh))){
     std::cout << (  mesh.empty()) << std::endl;
     std::cout << (  ( !CGAL::is_triangle_mesh(mesh))) << std::endl;
     std::cerr << "Input is not a triangle mesh." << std::endl;
     return 5;
     }
     */
    
    // create a property-map for SDF values
    typedef std::map<Polyhedron::Facet_const_handle, double> Facet_double_map;
    Facet_double_map internal_sdf_map;
    boost::associative_property_map<Facet_double_map> sdf_property_map(internal_sdf_map);
    // compute SDF values using default parameters for number of rays, and cone angle
    CGAL::sdf_values(mesh, sdf_property_map);
    
    char smoothing_lambda_str[5];
    snprintf(smoothing_lambda_str, sizeof(smoothing_lambda_str), "%.2f",smoothing_lambda );
    
    
    //print out the sdf values as well
    
    
    std::string output_filename_sdf = location_with_filename_str + "-cgal_" + patch::to_string(number_of_clusters) + "_"+smoothing_lambda_str + "_sdf.csv";
    //const char* output_filename_sdf_c = output_filename_sdf.c_str();
    std::ofstream myfile_sdf;
    myfile_sdf.open (output_filename_sdf.c_str());
    //myfile_sdf.open("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/auto_segmented_big_segments/output_sdf.csv");
    
    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
        facet_it != mesh.facets_end(); ++facet_it) {
        myfile_sdf << sdf_property_map[facet_it] << std::endl;
    }
    
    std::cout << std::endl;
    myfile_sdf.close();
    
    
    
    
    // create a property-map for segment-ids
    typedef std::map<Polyhedron::Facet_const_handle, std::size_t> Facet_int_map;
    Facet_int_map internal_segment_map;
    boost::associative_property_map<Facet_int_map> segment_property_map(internal_segment_map);
    // segment the mesh using default parameters for number of levels, and smoothing lambda
    // Any other scalar values can be used instead of using SDF values computed using the CGAL function
    
    
    
    //std::size_t number_of_clusters = clusters_array[j];       // use 4 clusters in soft clustering
    //double smoothing_lambda = smoothing_array[i];  // importance of surface features, suggested to be in-between [0,1]
    
    
    std::size_t number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);
    std::cout << "Number of segments: " << number_of_segments << std::endl;
    std::ofstream myfile;
    //create string for the float
    
    
    std::string output_filename = location_with_filename_str + "-cgal_" + patch::to_string(number_of_clusters) + "_"+smoothing_lambda_str;
    std::string output_filename_with_csv = output_filename + ".csv";
    //const char* output_filename_c = output_filename.c_str();
    myfile.open (output_filename_with_csv.c_str());
    //myfile.open("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/auto_segmented_big_segments/output.csv");
    // print segment-ids
    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
        facet_it != mesh.facets_end(); ++facet_it) {
        // ids are between [0, number_of_segments -1]
        //final_segments.push_back(std::to_string(segment_property_map[facet_it]) + ",");
        
        //going to write the segments to an output CSV file so that they can be loaded into blender later
        myfile << segment_property_map[facet_it] << std::endl;
        
        //myfile << "This is the first cell in the first column.\n";
        //myfile << "a,b,c,\n";
        //myfile << "c,s,v,\n";
    }
    
    std::cout << std::endl;
    myfile.close();
    
    
    
    // Note that we can use the same SDF values (sdf_property_map) over and over again for segmentation.
    // This feature is relevant for segmenting the mesh several times with different parameters.
    
    //CGAL::segmentation_from_sdf_values(
    //                                   mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);
    
    //need to save the segmentation values off here
    return 1;
}

int segmentation_example(){
    std::string off_file_data;
    
    std::string location = "/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/auto_segmented_big_segments/";
    std::string filename = "neuron_579228_Basal_2";
    std::string location_with_filename_str = location + filename;
    
    //"neuron_83286327_Basal_0" //200,000 faces
    //"neuron_579228_Basal_2" //29,000 file
    //2227952 2// 18,000
    //28571618 1 //largest
    
    
    
    
    
    //std::string files_to_process[2] = {"neuron_83286327_Basal_0" ,"neuron_579228_Basal_2" };
    //const int files_to_process_clusters[2] = {33,5};
    
    //std::string files_to_process[2] = { "neuron_28571618_Basal_1" ,"neuron_2227952_Basal_2" };
    //const int files_to_process_clusters[2] = {47,4};
    
    //std::string files_to_process[1] = {"neuron_83286327_Basal_0"};
    //const int files_to_process_clusters[1] = {24};
    //arrays that will be used for the loop
    
    
    /* for(int z=0;z<1;z = z+1){
     output_filename =  cgal_segmentation_vp2(location,files_to_process[z],files_to_process_clusters[z],0.04);
     std::cout << output_filename;
     }*/
    
    const double smoothing_array[1] = { 0.04};
    const int clusters_array[1] = {6};
    
    for( int j = 0; j< 1; j = j + 1){
        for( int i = 0; i < 1; i = i + 1 ){
            
            cgal_segmentation(location_with_filename_str.c_str(),clusters_array[j],smoothing_array[i]);
        }
    }
    
    return 100;
    //std::cout << output_filename;
    
    
};

static PyObject* cgal_segmentation_C(PyObject *self, PyObject *args)
{
    int number_of_clusters;
    double smoothing_lambda;
    const char *location_with_filename;
    int output_int;
    
    if (!PyArg_ParseTuple(args,"sid",&location_with_filename,&number_of_clusters,&smoothing_lambda))
        return NULL;
    
    output_int = cgal_segmentation(location_with_filename,number_of_clusters,smoothing_lambda);
    return Py_BuildValue("i",output_int);
    
}


//has a problem with how this only accepts one aragument
static PyObject* cgal_demo(PyObject *self){
    return Py_BuildValue("i",segmentation_example());
}


static PyMethodDef cgal_Segmentation_Methods[] =
{
    { "cgal_segmentation", cgal_segmentation_C, METH_VARARGS, "calculates the mesh segmentation" },
    { "cgal_demo", (PyCFunction)cgal_demo, METH_NOARGS, "runs an example segmentation" },
    { NULL,NULL,0, NULL }
    
};

static struct PyModuleDef cgal_Segmentation_Module =
{
    PyModuleDef_HEAD_INIT,
    "cgal_Segmentation_Module",
    "CGAL Neuron Segmentation Module",
    -1,
    cgal_Segmentation_Methods
};

PyMODINIT_FUNC PyInit_cgal_Segmentation_Module(void )
{
    return PyModule_Create(&cgal_Segmentation_Module);
}




