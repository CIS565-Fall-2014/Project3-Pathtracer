// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
// Edited by Liam Boone for use with CUDA v5.5

#include <iostream>
#include "scene.h"
#include <cstring>
#include "tiny_obj_loader.h"

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
			}
		}
	}
}


//load obj file, returns the number of triangle faces
int loadMeshFromOBJ(std::vector<tinyobj::shape_t>& shapes, std::vector<tinyobj::material_t>& materials, const char* filename,
	std::vector<triangle>& tris, std::vector<glm::vec3>& BB){   //thanks to TinyObjLoader @ https://github.com/syoyo/tinyobjloader

	std::string error = tinyobj::LoadObj(shapes,materials, filename); //tiny obj loader

	if (error.length()>0) {
		std::cerr << error << std::endl;
		return 0;
	}
	//there is only one shape in our situation
	int length = shapes[0].mesh.indices.size();
	int numofTris = length /3;
	std::vector<unsigned int> idx = shapes[0].mesh.indices;   //indice array
	std::vector<float> pos = shapes[0].mesh.positions;   //position array
	for(int i = 0; i < length; i+=3){
		int i1 = idx[i];
		int i2 = idx[i+1];
		int i3 = idx[i+2];

		glm::vec3 p1 = glm::vec3(pos[3 * i1 + 0], pos[3 * i1 + 1], pos[3 * i1 + 2]);				
		glm::vec3 p2 = glm::vec3(pos[3 * i2 + 0], pos[3 * i2 + 1], pos[3 * i2 + 2]);
		glm::vec3 p3 = glm::vec3(pos[3 * i3 + 0], pos[3 * i3 + 1], pos[3 * i3 + 2]);
		//glm::vec3 edge12 = p2 - p1;
		//glm::vec3 edge13 = p3 - p1;
		triangle tri;
		tri.p1 =  p1;
		tri.p2 =  p2;
		tri.p3 =  p3;
		tris.push_back ( tri );
	}

	//demtermine bounding box for the speed of intersection test
	float xMin = FLT_MAX, xMax = FLT_MIN, yMin = FLT_MAX, yMax = FLT_MIN, zMin = FLT_MAX, zMax = FLT_MIN;
	for(int k = 0; k < numofTris; k++){
		xMin = min( xMin, tris[k].p1.x );
		xMin = min( xMin, tris[k].p2.x );
		xMin = min( xMin, tris[k].p3.x );

		xMax = max( xMax, tris[k].p1.x );
		xMax = max( xMax, tris[k].p2.x );
		xMax = max( xMax, tris[k].p3.x );

		yMin = min( yMin, tris[k].p1.y );
		yMin = min( yMin, tris[k].p2.y );
		yMin = min( yMin, tris[k].p3.y );

		yMax = max( yMax, tris[k].p1.y );
		yMax = max( yMax, tris[k].p2.y );
		yMax = max( yMax, tris[k].p3.y );

		zMin = min( zMin, tris[k].p1.z );
		zMin = min( zMin, tris[k].p2.z );
		zMin = min( zMin, tris[k].p3.z );

		zMax = max( zMax, tris[k].p1.z );
		zMax = max( zMax, tris[k].p2.z );
		zMax = max( zMax, tris[k].p3.z );
	}
	BB.push_back( glm::vec3( xMin, yMin, zMin ) );  
	BB.push_back( glm::vec3( xMax, yMax, zMax ) );

	/*	printf("BB:\n [%f, %f, %f]\n [%f, %f, %f]\n", 
			BB[0].x,  BB[0].y,  BB[0].z, 
			BB[1].x, BB[1].y,  BB[1].z);

	for(int k=0; k<length/3; k++){
		printf("points:\n [%f, %f, %f]\n [%f, %f, %f]\n [%f, %f, %f]\n", 
			tris[k].p1.x,  tris[k].p1.y,  tris[k].p1.z, 
			tris[k].p2.x, tris[k].p2.y,  tris[k].p2.z, 
			tris[k].p3.x,  tris[k].p3.y,  tris[k].p3.z);
	}*/
	return numofTris;
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
					std::vector<tinyobj::shape_t> shapes;
					std::vector<tinyobj::material_t> materials;
					std::vector<triangle> triangles;
					std::vector<glm::vec3> bb;
					int num = loadMeshFromOBJ(shapes, materials, line.c_str(), triangles, bb);
					if( num> 0){
						newObject.type = MESH;
						newObject.numOfTris = num;
						newObject.tris = new triangle[num];
						for(int i=0; i<num; i++){
							newObject.tris[i] = triangles[i];
						}
					/*	for (int i=0; i<num; i++){   //for each triangle
							printf("triangle: \n [%.2f, %.2f, %.2f]\n [%.2f, %.2f, %.2f]\n [%.2f, %.2f, %.2f]\n",
								newObject.tris[i].p1.x, newObject.tris[i].p1.y, newObject.tris[i].p1.z,
								newObject.tris[i].p2.x, newObject.tris[i].p2.y, newObject.tris[i].p2.z,
								newObject.tris[i].p3.x, newObject.tris[i].p3.y, newObject.tris[i].p3.z);
						}*/
						newObject.bBoxMax = bb[1];
						newObject.bBoxMin = bb[0];
						/*printf("bb:\n [%f, %f, %f]\n [%f, %f, %f]\n", 
									newObject.bBoxMin.x,  newObject.bBoxMin.y, newObject.bBoxMin.z, 
									newObject.bBoxMax.x, newObject.bBoxMax.y,  newObject.bBoxMax.z);*/
					}
					else{
						 cout << "ERROR: " << " cannot load obj file: " << line << endl;
						 return -1;
					}
                }else{
                    cout << "ERROR: " << line << " is not a valid object type!" << endl;
                    return -1;
                }

            }
        }
       
	//link material
    utilityCore::safeGetline(fp_in,line);
	if(!line.empty() && fp_in.good()){
	    vector<string> tokens = utilityCore::tokenizeString(line);
	    newObject.materialid = atoi(tokens[1].c_str());
	    cout << "Connecting Object " << objectid << " to Material " << newObject.materialid << "..." << endl;
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
	int num;
	if(DEPTH_OF_FIELD){
		num = 6;  //6 lines to read
	}
	else{
		num = 4;
	}
	for(int i=0; i<num; i++){
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
		}else if(strcmp(tokens[0].c_str(), "FOCAL")==0){
			newCamera.focalLength = atof(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "APERTURE")==0){
			newCamera.aperture = atof(tokens[1].c_str());
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
