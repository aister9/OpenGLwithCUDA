#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <iostream>

bool PointInTriangle(glm::vec3 point, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3);
class OBJReader {
public:
	std::vector < glm::vec3 > vertexs;
	std::vector < glm::vec3 > normals;
	std::vector < GLuint > faces;
	std::vector < GLuint > facesnormal;

	//In this version only read vertex data
	void readObj(std::string filePath) {
		std::ifstream mfile(filePath);
		if (mfile.is_open()) {
			while (!mfile.eof()) {
				std::string s;
				mfile >> s;

				if (s._Equal("v")) {
					float x, y, z;
					std::string nss;
					mfile >> nss;
					x = atof(nss.c_str());
					mfile >> nss;
					y = atof(nss.c_str());
					mfile >> nss;
					z = atof(nss.c_str());

					glm::vec3 newV(x, y, z);
					vertexs.push_back(newV);
				}
				else if (s._Equal("f")) { // faces
					for (int i = 0; i < 3; i++) {
						std::string nss;
						mfile >> nss;
						int pos = nss.find('/');
						std::string value = nss.substr(0, pos);
						int number = atoi(value.c_str());
						faces.push_back(number);
					}
				}
			}
		}
		else {
			std::cout << "can`t open file" << std::endl;
		}

		mfile.close();
	}

	void saveFileToPCD(std::string filePath) {
		std::ofstream pcdFile(filePath);

		if (!pcdFile.is_open()) return;

		pcdFile << "VERSION .7" << std::endl;
		pcdFile << "FIELDS x y z" << std::endl;
		pcdFile << "SIZE 4 4 4" << std::endl;
		pcdFile << "TYPE F F F" << std::endl;
		pcdFile << "COUNT 1 1 1" << std::endl;
		pcdFile << "WIDTH " << vertexs.size() << std::endl;
		pcdFile << "HEIGHT 1" << std::endl;
		pcdFile << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
		pcdFile << "POINTS " << vertexs.size() << std::endl;
		pcdFile << "DATA ascii" << std::endl;

		for (glm::vec3 vertexd : vertexs) {
			pcdFile << vertexd.x << " " << vertexd.y << " " << vertexd.z << std::endl;
		}

		pcdFile.close();
	}

	void vertexNormalize() {
		glm::vec3 max_xyz(0, 0, 0);
		glm::vec3 min_xyz(0, 0, 0);

		for (glm::vec3 vertexd : vertexs) {
			max_xyz.x = (max_xyz.x < vertexd.x) ? vertexd.x : max_xyz.x;
			max_xyz.y = (max_xyz.y < vertexd.y) ? vertexd.y : max_xyz.y;
			max_xyz.z = (max_xyz.z < vertexd.z) ? vertexd.z : max_xyz.z;

			min_xyz.x = (min_xyz.x > vertexd.x) ? vertexd.x : min_xyz.x;
			min_xyz.y = (min_xyz.y > vertexd.y) ? vertexd.y : min_xyz.y;
			min_xyz.z = (min_xyz.z > vertexd.z) ? vertexd.z : min_xyz.z;
		}

		float maxDiff = std::max({ max_xyz.x - min_xyz.x, max_xyz.y - min_xyz.y,max_xyz.z - min_xyz.z });
		glm::vec3 average_xyz = (max_xyz + min_xyz) / 2.f;

		for (glm::vec3& vertexd : vertexs) {
			vertexd.x = (vertexd.x - average_xyz.x) / maxDiff;
			vertexd.y = (vertexd.y - average_xyz.y) / maxDiff;
			vertexd.z = (vertexd.z - average_xyz.z) / maxDiff;
		}
	}

	void vertexAdd(int genNum) {
		std::random_device rd;
		std::mt19937 gen(rd());

		for (int i = 0; i < faces.size(); i++) {
			glm::vec3 v1 = vertexs[faces[i*3+0]];
			glm::vec3 v2 = vertexs[faces[i*3+1]];
			glm::vec3 v3 = vertexs[faces[i*3+2]];

			std::uniform_real_distribution<float> r1(0, 1);
			std::uniform_real_distribution<float> r2(0, 1);

			int samples = 0;
			while (samples < genNum) {
				float alpha = r1(gen), beta = r2(gen);
				glm::vec3 newPoint = (1 - sqrt(alpha)) * v1 + (sqrt(alpha) * (1 - beta)) * v2 + (beta * sqrt(alpha)) * v3;

				if (PointInTriangle(newPoint, v1, v2, v3)) {
					samples++;
					vertexs.push_back(newPoint);
				}
				else continue;
			}
		}
	}
};

bool PointInTriangle(glm::vec3 point, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {
	glm::vec3 a = t1 - point;
	glm::vec3 b = t2 - point;
	glm::vec3 c = t3 - point;

	glm::vec3 u = glm::cross(b, c);
	glm::vec3 v = glm::cross(c, a);
	glm::vec3 w = glm::cross(a, b);

	if (glm::dot(u, v) < 0.f) return false;
	if (glm::dot(u, w) < 0.f) return false;

	return true;
}
