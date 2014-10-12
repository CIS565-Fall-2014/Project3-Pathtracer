// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
// Edited by Liam Boone for use with CUDA v5.5

#include <iostream>
#include "scene.h"
#include <cstring>

scene::scene(string filename){
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
		while(fp_in.good()){
			string line;
            utilityCore::safeGetline(fp_in,line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "MATERIAL")==0){
				    loadMaterial(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "OBJECT")==0){
				    loadObject(tokens[1]);
				    cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "CAMERA")==0){
				    loadCamera();
				    cout << " " << endl;
				}
				else if ( strcmp( tokens[0].c_str(), "TEXTURE" ) == 0 ) {
					loadTextures( tokens[1] );
					std::cout << " " << std::endl;
				}
			}
		}
	}
}

int scene::loadObject(string objectid){
    int id = atoi(objectid.c_str());
    if(id!=objects.size()){
        cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
        return -1;
    }else{
        cout << "Loading Object " << id << "..." << endl;
        geom newObject;
        string line;
        
        //load object type 
        utilityCore::safeGetline(fp_in,line);
        if (!line.empty() && fp_in.good()){
            if(strcmp(line.c_str(), "sphere")==0){
                cout << "Creating new sphere..." << endl;
				newObject.type = SPHERE;
            }else if(strcmp(line.c_str(), "cube")==0){
                cout << "Creating new cube..." << endl;
				newObject.type = CUBE;
            }else{
				string objline = line;
                string name;
                string extension;
                istringstream liness(objline);
                getline(liness, name, '.');
                getline(liness, extension, '.');
                if(strcmp(extension.c_str(), "obj")==0){
                    cout << "Creating new mesh..." << endl;
                    cout << "Reading mesh from " << line << "... " << endl;
		    		newObject.type = MESH;
                }else{
                    cout << "ERROR: " << line << " is not a valid object type!" << endl;
                    return -1;
                }
            }
        }
       
	//link material
    utilityCore::safeGetline(fp_in,line);
	if(!line.empty() && fp_in.good()) {
	    vector<string> tokens = utilityCore::tokenizeString(line);
	    newObject.materialid = atoi(tokens[1].c_str());
	    cout << "Connecting Object " << objectid << " to Material " << newObject.materialid << "..." << endl;
	}

	// Attach textures.
	utilityCore::safeGetline( fp_in,line );
	if( !line.empty() && fp_in.good() ) {
		vector<string> tokens = utilityCore::tokenizeString( line );
		if ( strcmp( tokens[0].c_str(), "texture" ) == 0 ) {
			newObject.texture_id = atoi( tokens[1].c_str() );
			cout << "Loading texture " << newObject.texture_id << " for object " << objectid << "..." << endl;
		}
	}
        
	//load frames
    int frameCount = 0;
    utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> translations;
	vector<glm::vec3> scales;
	vector<glm::vec3> rotations;
    while (!line.empty() && fp_in.good()){
	    
	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
        if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
            cout << "ERROR: Incorrect frame count!" << endl;
            return -1;
        }
	    
	    //load tranformations
	    for(int i=0; i<3; i++){
            glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in,line);
            tokens = utilityCore::tokenizeString(line);
            if(strcmp(tokens[0].c_str(), "TRANS")==0){
                translations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "ROTAT")==0){
                rotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "SCALE")==0){
                scales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }
	    }
	    
	    frameCount++;
        utilityCore::safeGetline(fp_in,line);
	}
	
	//move frames into CUDA readable arrays
	newObject.translations = new glm::vec3[frameCount];
	newObject.rotations = new glm::vec3[frameCount];
	newObject.scales = new glm::vec3[frameCount];
	newObject.transforms = new cudaMat4[frameCount];
	newObject.inverseTransforms = new cudaMat4[frameCount];
	for(int i=0; i<frameCount; i++){
		newObject.translations[i] = translations[i];
		newObject.rotations[i] = rotations[i];
		newObject.scales[i] = scales[i];
		glm::mat4 transform = utilityCore::buildTransformationMatrix(translations[i], rotations[i], scales[i]);
		newObject.transforms[i] = utilityCore::glmMat4ToCudaMat4(transform);
		newObject.inverseTransforms[i] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	}
	
        objects.push_back(newObject);
	
	cout << "Loaded " << frameCount << " frames for Object " << objectid << "!" << endl;
        return 1;
    }
}

int scene::loadCamera(){
	cout << "Loading Camera ..." << endl;
        camera newCamera;
	float fovy;
	
	//load static properties
	for(int i=0; i<4; i++){
		string line;
        utilityCore::safeGetline(fp_in,line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "RES")==0){
			newCamera.resolution = glm::vec2(atoi(tokens[1].c_str()), atoi(tokens[2].c_str()));
		}else if(strcmp(tokens[0].c_str(), "FOVY")==0){
			fovy = atof(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "ITERATIONS")==0){
			newCamera.iterations = atoi(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "FILE")==0){
			newCamera.imageName = tokens[1];
		}
	}
        
	//load time variable properties (frames)
    int frameCount = 0;
	string line;
    utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> positions;
	vector<glm::vec3> views;
	vector<glm::vec3> ups;
    while (!line.empty() && fp_in.good()){
	    
	    //check frame number
	    vector<string> tokens = utilityCore::tokenizeString(line);
        if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
            cout << "ERROR: Incorrect frame count!" << endl;
            return -1;
        }
	    
	    //load camera properties
	    for(int i=0; i<3; i++){
            //glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in,line);
            tokens = utilityCore::tokenizeString(line);
            if(strcmp(tokens[0].c_str(), "EYE")==0){
                positions.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "VIEW")==0){
                views.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }else if(strcmp(tokens[0].c_str(), "UP")==0){
                ups.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
            }
	    }
	    
	    frameCount++;
        utilityCore::safeGetline(fp_in,line);
	}
	newCamera.frames = frameCount;
	
	//move frames into CUDA readable arrays
	newCamera.positions = new glm::vec3[frameCount];
	newCamera.views = new glm::vec3[frameCount];
	newCamera.ups = new glm::vec3[frameCount];
	for(int i = 0; i < frameCount; i++){
		newCamera.positions[i] = positions[i];
		newCamera.views[i] = views[i];
		newCamera.ups[i] = ups[i];
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy*(PI/180));
	float xscaled = (yscaled * newCamera.resolution.x)/newCamera.resolution.y;
	float fovx = (atan(xscaled)*180)/PI;
	newCamera.fov = glm::vec2(fovx, fovy);

	renderCam = newCamera;
	
	//set up render camera stuff
	renderCam.image = new glm::vec3[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	renderCam.rayList = new ray[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	for(int i=0; i<renderCam.resolution.x*renderCam.resolution.y; i++){
		renderCam.image[i] = glm::vec3(0,0,0);
	}
	
	cout << "Loaded " << frameCount << " frames for camera!" << endl;
	return 1;
}

int scene::loadMaterial(string materialid){
	int id = atoi(materialid.c_str());
	if(id!=materials.size()){
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}else{
		cout << "Loading Material " << id << "..." << endl;
		material newMaterial;
	
		//load static properties
		for(int i=0; i<10; i++){
			string line;
            utilityCore::safeGetline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "RGB")==0){
				glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.color = color;
			}else if(strcmp(tokens[0].c_str(), "SPECEX")==0){
				newMaterial.specularExponent = atof(tokens[1].c_str());				  
			}else if(strcmp(tokens[0].c_str(), "SPECRGB")==0){
				glm::vec3 specColor( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.specularColor = specColor;
			}else if(strcmp(tokens[0].c_str(), "REFL")==0){
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFR")==0){
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFRIOR")==0){
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "SCATTER")==0){
				newMaterial.hasScatter = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "ABSCOEFF")==0){
				glm::vec3 abscoeff( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.absorptionCoefficient = abscoeff;
			}else if(strcmp(tokens[0].c_str(), "RSCTCOEFF")==0){
				newMaterial.reducedScatterCoefficient = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "EMITTANCE")==0){
				newMaterial.emittance = atof(tokens[1].c_str());					  
			
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}




const std::string TEXTURE_PATH = "C:\\Users\\Danny\\Documents\\_projects\\cis565\\Project3-Pathtracer\\data\\textures\\";


//glm::vec2 extractTextureDimensions( std::string filename )
//{
//	FILE *f = fopen( filename.c_str(), "rb" );
//	unsigned char info[54];
//	fread( info, sizeof( unsigned char ), 54, f ); // Read the 54-byte header.
//
//	// Extract image height and width from header.
//	int width = *( int* )&info[18];
//	int height = *( int* )&info[22];
//
//	fclose( f );
//
//	return glm::vec2( width, height );
//}


//void BMPToImage( const std::string &filename, image &texture_image )
//{
//	int i;
//	FILE *f = fopen( filename.c_str(), "rb" );
//	unsigned char info[54];
//	fread( info, sizeof( unsigned char ), 54, f ); // read the 54-byte header
//
//	// extract image height and width from header
//	int width = *( int* )&info[18];
//	int height = *( int* )&info[22];
//
//	int size = 3 * width * height;
//	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
//	fread( data, sizeof( unsigned char ), size, f ); // read the rest of the data at once
//	fclose( f );
//
//	for ( i = 0; i < size; i += 3 ) {
//		unsigned char tmp = data[i];
//		data[i] = data[i+2];
//		data[i+2] = tmp;
//	}
//
//	//for ( i = 0; i < size; i += 3 ) {
//	//	int x = i % width;
//	//	int y = i / width;
//	//	glm::vec3 rgb( data[i], data[i + 1], data[i + 2] );
//	//	texture_image.writePixelRGB( x, y, rgb );
//	//}
//
//	// DEBUG.
//	//std::cout << "Texture dimensions: ( " << width << ", " << height << " )" << std::endl;
//	//std::cin.ignore();
//
//	for ( int y = 0; y < height; ++y ) {
//		for ( int x = 0; x < width; ++x ) {
//			int buffer_loc = ( ( y * width ) + x ) * 3;
//			glm::vec3 rgb( data[buffer_loc] / 255.0f, data[buffer_loc + 1] / 255.0f, data[buffer_loc + 2] / 255.0f );
//			texture_image.writePixelRGB( x, y, rgb );
//
//			// DEBUG.
//			//std::cout << "( x, y ) = ( " << x << ", " << y << " )" << std::endl;
//			//std::cout << "buffer_loc = " << buffer_loc << std::endl;
//			//std::cout << "Writing RGB for pixel ( " << x << ", " << y << " ): ( " << rgb.x << ", " << rgb.y << ", " << rgb.z << " )" << std::endl;
//			//std::cin.ignore();
//		}
//
//		// DEBUG
//		//std::cin.ignore();
//	}
//
//	// DEBUG
//	//std::cin.ignore();
//}


//int scene::loadTextures( std::string texture_id )
//{
//	// DEBUG
//	//std::cout << "Inside int scene::loadTextures( std::string texture_id )." << std::endl;
//
//	int id = atoi( texture_id.c_str() );
//	
//	if ( id != textures.size() ) {
//		std::cout << "ERROR: TEXTURE ID does not match expected number of textures." << std::endl;
//		return -1;
//	}
//	else{
//		cout << "Loading texture " << id << "..." << endl;
//
//		std::string line;
//        utilityCore::safeGetline( fp_in, line );
//		std::vector<std::string> tokens = utilityCore::tokenizeString( line );
//
//		std::string texture_filename = "";
//		if ( strcmp( tokens[0].c_str(), "FILE" ) == 0 ) {
//			texture_filename = tokens[1];
//		}
//
//		std::string texture_full_path = TEXTURE_PATH + texture_filename;
//		glm::vec2 texture_dimensions = extractTextureDimensions( texture_full_path );
//
//		// DEBUG.
//		//std::cout << "Texture dimensions: ( " << texture_dimensions.x << ", " << texture_dimensions.y << " )" << std::endl;
//
//		image texture_image( texture_dimensions.x, texture_dimensions.y );
//		BMPToImage( texture_full_path, texture_image );
//	
//		textures.push_back( texture_image );
//		return 1;
//	}
//}


int scene::loadTextures( std::string texture_id )
{
	int id = atoi( texture_id.c_str() );
	
	if ( id != textures.size() ) {
		std::cout << "ERROR: TEXTURE ID does not match expected number of textures." << std::endl;
		return -1;
	}
	else{
		cout << "Loading texture " << id << "..." << endl;

		std::string line;
        utilityCore::safeGetline( fp_in, line );
		std::vector<std::string> tokens = utilityCore::tokenizeString( line );

		std::string texture_filename = "";
		if ( strcmp( tokens[0].c_str(), "FILE" ) == 0 ) {
			texture_filename = tokens[1];
		}

		std::string texture_full_path = TEXTURE_PATH + texture_filename;

		int i;
		FILE *f = fopen( texture_full_path.c_str(), "rb" );
		unsigned char info[54];
		fread( info, sizeof( unsigned char ), 54, f ); // read the 54-byte header

		// extract image height and width from header
		int width = *( int* )&info[18];
		int height = *( int* )&info[22];

		// Currently, textures are limited to images of at most 1024x1024 resolution.
		// The number of pixels is hard-coded in the texture struct in sceneStructs.h.
		if ( width > 1024 || height > 1024 ) {
			std::cout << "ERROR: Texture is too large: ( " << width << ", " << height << " )" << std::endl;
			return -1;
		}

		//image texture_image( width, height );

		int size = 3 * width * height;
		unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
		fread( data, sizeof( unsigned char ), size, f ); // read the rest of the data at once
		fclose( f );

		//for ( i = 0; i < size; i += 3 ) {
		//	unsigned char tmp = data[i];
		//	data[i] = data[i+2];
		//	data[i+2] = tmp;
		//}

		simpleTexture texture_image;
		texture_image.dimensions = glm::vec2( width, height );

		for ( int y = 0; y < height; ++y ) {
			for ( int x = 0; x < width; ++x ) {
				int linear_index = ( y * width ) + x;
				int bmp_buffer_loc = linear_index * 3;
				glm::vec3 rgb( data[bmp_buffer_loc + 1] / 255.0f, data[bmp_buffer_loc] / 255.0f, data[bmp_buffer_loc + 2] / 255.0f );
				//texture_image.writePixelRGB( x, y, rgb );
				texture_image.rgb[linear_index] = rgb;

				// DEBUG.
				//std::cout << "RGB for index " << linear_index << ": ( " << rgb.x << ", " << rgb.y << ", " << rgb.z << " )" << std::endl;
			}
		}

		// DEBUG.
		//std::cin.ignore();

		textures.push_back( texture_image );
		return 1;
	}
}