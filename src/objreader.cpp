#include "objreader.h"

//vv for vertices positions
//vi for each triangle point's index
//fn for each triangle's normal
void OBJreader(vector<vec3> &vv,vector<vec3> &fn,vector<vec3> &vi,string file,vec3 &maxpoint,vec3 &minpoint)
{	
	vi.clear();
	vv.clear();
	fn.clear();
	string line;
	ifstream infile(file,ios::in);
	vector<vec3> vn;
	vector<vec3> fnindex;

	float tempmax0,tempmax1,tempmax2,tempmin0,tempmin1,tempmin2;
	maxpoint[0]=-1e20f;
	minpoint[0]=1e20f;
	maxpoint[1]=-1e20f;
	minpoint[1]=1e20f;
	maxpoint[2]=-1e20f;
	minpoint[2]=1e20f;

	/*******************/


	while(getline(infile, line))
	{
		if (line.substr(0,2) == "v ") 
		{
			istringstream s(line.substr(2));
			vec3 v; 
			s >> v[0]; s >> v[1]; s >> v[2];
			v = v;
			vv.push_back(v); 
			tempmax0=(float)v[0];
			tempmin0=(float)v[0];
			tempmax1=(float)v[1];
			tempmin1=(float)v[1];
			tempmax2=(float)v[2];
			tempmin2=(float)v[2];
			maxpoint[0]=std::max((float)maxpoint[0],tempmax0);
			minpoint[0]=std::min((float)minpoint[0],tempmin0);
			maxpoint[1]=std::max((float)maxpoint[1],tempmax1);
			minpoint[1]=std::min((float)minpoint[1],tempmin1);
			maxpoint[2]=std::max((float)maxpoint[2],tempmax2);
			minpoint[2]=std::min((float)minpoint[2],tempmin2);
		}
		else if (line.substr(0,3) == "vn ") 
		{
			istringstream s(line.substr(3));
			vec3 v; 
			s >> v[0]; s >> v[1]; s >> v[2];
			v = v;
			vn.push_back(v); 
		}
		else if (line.substr(0,2) == "f ") 
		{
			istringstream s(line.substr(2));
			vec3 v; 
			vec3 vf; 
			string a,b,c;
			string tempa,tempb,tempc;
			int cut;
			s >> a; s >> b; s >> c; 
			for(int i=0;i<(int)a.length();i++)
			{
				if(a.at(i)=='/')
				{
					cut=i;
					tempa=a.substr(0,cut);
					break;
				}

				if(i==(int)a.length()-1)
				{
					cut=i;
					tempa=a.substr(0,cut+1); 
				}
			}

			for(int i=0;i<(int)b.length();i++)
			{
				if(b.at(i)=='/')
				{
					cut=i;
					tempb=b.substr(0,cut);
					break;
				}

				if(i==(int)b.length()-1)
				{
					cut=i;
					tempb=b.substr(0,cut+1); 
				}
			}

			for(int i=0;i<(int)c.length();i++)
			{
				if(c.at(i)=='/')
				{
					cut=i;
					tempc=c.substr(0,cut);
					break;
				}

				if(i==(int)c.length()-1)
				{
					cut=i;
					tempc=c.substr(0,cut+1); 
				}
			}


			v[0]=atoi(tempa.c_str());
			v[1]=atoi(tempb.c_str());
			v[2]=atoi(tempc.c_str());

			v[0]--;
			v[1]--;
			v[2]--;

			vi.push_back(v);

			if(vn.size()!=0)
			{
				//vnormal index
				for(int i=(int)a.length()-1;i>=0;--i)
				{
					if(a.at(i)=='/')
					{
						cut=i;
						tempa=a.substr(i+1,(int)a.length());
						break;
					}
				}

				for(int i=(int)b.length()-1;i>=0;--i)
				{
					if(b.at(i)=='/')
					{
						cut=i;
						tempb=b.substr(i+1,(int)b.length());
						break;
					}
				}

				for(int i=(int)c.length()-1;i>=0;--i)
				{
					if(c.at(i)=='/')
					{
						cut=i;
						tempc=c.substr(i+1,(int)c.length());
						break;
					}
				}

				v[0]=atoi(tempa.c_str());
				v[1]=atoi(tempb.c_str());
				v[2]=atoi(tempc.c_str());

				v[0]--;
				v[1]--;
				v[2]--;
				fnindex.push_back(v);
			}
		}
	}

	//Scale all obj files to the size of 1(bounding box's longest edge is 1)
	float x = maxpoint[0]-minpoint[0];
	float y = maxpoint[1]-minpoint[1];
	float z = maxpoint[2]-minpoint[2];
	float maxdim = std::max(std::max(x,y),z);
	x = x/maxdim;
	y = y/maxdim;
	z = z/maxdim;
	for(int i=0;i<(int)vv.size();i++)
	{
		vv[i] = vv[i]-minpoint;
		vv[i][0] = (vv[i][0]/(maxpoint[0]-minpoint[0]) - 0.5f) * x;
		vv[i][1] = (vv[i][1]/(maxpoint[1]-minpoint[1]) - 0.5f) * y;
		vv[i][2] = (vv[i][2]/(maxpoint[2]-minpoint[2]) - 0.5f) * z;
	}

	//for container
	maxpoint[0] = 0.5f * x;
	maxpoint[1] = 0.5f * y;
	maxpoint[2] = 0.5f * z;
	minpoint[0] = -0.5f * x;
	minpoint[1] = -0.5f * y;
	minpoint[2] = -0.5f * z;


	//If the obj file has vn
	if(vn.size()!=0)
	{
		for(unsigned int i = 0; i < fnindex.size() ; i++)
		{
			vec3 n1 = vn[fnindex[i][0]];
			vec3 n2 = vn[fnindex[i][1]];
			vec3 n3 = vn[fnindex[i][2]];
			vec3 fnormal = (n1+n2+n3)/3.0f;
			fnormal = normalize(fnormal);
			fn.push_back(fnormal);
		}
	}
	else   //compute face normal by myself
	{
	   cout<<file<<" has no vn, compute face normal, the result may be wrong if the vertices are not ordered!"<<endl;
	   for(unsigned int i = 0; i < vi.size() ; i++)
	   {
		   vec3 fnormal = normalize(cross(vv[vi[i][1]]-vv[vi[i][0]],vv[vi[i][2]]-vv[vi[i][0]]));
		   fn.push_back(fnormal);
	   }
	}
}
