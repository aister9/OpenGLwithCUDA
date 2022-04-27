#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cublas.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>

#include "OpenGLSupport.h"
#include "objReader.h"

using namespace std;

OBJReader reader;

bool glfwewInit(GLFWwindow **window, int width, int height) {
    if (!glfwInit()) return false; // glfw 초기화

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    *(window) = glfwCreateWindow(width, height, "Test window", nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        cout << "failed" << endl;
        return false;
    }

    glfwMakeContextCurrent(*window); // 윈도우 컨텍스트 생성

    glfwSwapInterval(0); // 스왑 간격 : 0 설정하면 fps 제한X, 1 설정하면 fps 제한 60

    if (glewInit() != GLEW_OK) { // GLEW 초기호 실패하면 종료
        glfwTerminate();
        return false;
    }

    cout << glGetString(GL_VERSION) << endl; // OpenGL 버전

    return true;
}
void drawLoop(GLFWwindow* window, GLuint programID, GLuint vertexbuffer, vector<glm::vec3> points);
void drawLoop(GLFWwindow* window, GLuint programID, GLuint vertexbuffer, vector<glm::vec3> points, GLuint projID, float *m2c, GLuint cam_k_ID, float *K);

vector<glm::vec3> projection(vector<glm::vec3> origin, float *m2c, float *K);
vector<glm::vec3> projection_parallel(vector<glm::vec3> origin, float *m2c, float *K);
vector<glm::vec3> cudaProjection(vector<glm::vec3> origin, float* m2c, float* k);

__global__ void projection_cuda(float* origin, float* target, int size, float* m2c, float* K);

int main()
{
    GLFWwindow* window;
    int width = 640, height = 480;

    reader.readObj("Duck.obj");
    cout << "read obj complete. number of vertex : " << reader.vertexs.size() << "" << endl;

    if (!glfwewInit(&window, width, height)) return -1;

    float cam_k[3][3]{ {572.4114, 0.0, 325.2611},{0.0, 573.57043, 242.04899},{0.0, 0.0, 1.0} };
    float cam_m2c[3][4]{ {0.99555397, -0.02460450, 0.09092280, 65.21082500}, {0.03204600, -0.81922501, -0.57257599, 30.60059019}, {0.08857420, 0.57294399, -0.81479502, 822.65398262} };

    auto current_time = []() {auto t = chrono::system_clock::now(); return chrono::duration_cast<chrono::nanoseconds>(t.time_since_epoch()).count(); };
    auto start = current_time();
    vector<glm::vec3> proj = projection(reader.vertexs, &cam_m2c[0][0], &cam_k[0][0]);
    auto end = current_time();

    cout << "host projection time(ms) : " << (end - start)/1000000. << endl;

    start = current_time();
    vector<glm::vec3> proj_omp = projection_parallel(reader.vertexs, &cam_m2c[0][0], &cam_k[0][0]);
    end = current_time();

    cout << "host projection(omp) time(ms) : " << (end - start) / 1000000. << endl;

    vector<glm::vec3> proj_cuda = cudaProjection(reader.vertexs, &cam_m2c[0][0], &cam_k[0][0]);

    /*for (int i = 0; i < proj.size()*3; i++) {
        if ((*proj.data())[i] - (*proj_omp.data())[i] > 0.0000001f) {
            cout << "error is not equal proj and proj_omp" << endl;
            break;
        }
        if ((*proj.data())[i] - (*proj_cuda.data())[i] > 0.0000001f) {
            cout << "error is not equal proj and proj_cuda" << endl;
            break;
        }
    }*/
    //cublasInit();

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * proj_cuda.size(), &proj_cuda[0], GL_STATIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * reader.vertexs.size(), &reader.vertexs[0], GL_STATIC_DRAW);

    //GLuint edgebuffer;
    //glGenBuffers(1, &edgebuffer);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgebuffer);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * reader.faces.size(), &reader.faces[0], GL_STATIC_DRAW);

    //GLuint programID = LoadShaders("n_vertex.shader", "n_frag.shader");
    GLuint programID = LoadShaders("proj_vertex.shader", "n_frag.shader");

    //if projection in shader
    GLuint m2cID = glGetUniformLocation(programID, "m2c");
    GLuint KID = glGetUniformLocation(programID, "K");

    double lastTime = glfwGetTime();
    int nbFrames{ 0 };
    int loop = 0;
    while (!glfwWindowShouldClose(window) && loop<10) {
        //drawLoop(window, programID, vertexbuffer, proj);
        drawLoop(window, programID, vertexbuffer, reader.vertexs, m2cID, &cam_m2c[0][0], KID, &cam_k[0][0]);

        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) {
            printf("%f ms/frame, FPS = %d\n", 1000.0 / (double)nbFrames, nbFrames);
            nbFrames = 0;
            lastTime += 1.0;
            loop++;
        }
    }

    return 0;
}


vector<glm::vec3> projection(vector<glm::vec3> origin, float* m2c, float* K) {
    vector<glm::vec3> res;

    for (glm::vec3 v : origin) {
        glm::vec3 temp(v);

        //multiply m2c
        temp.x = m2c[0 * 4 + 0] * v.x + m2c[0 * 4 + 1]*v.y + m2c[0 * 4 + 2]*v.z + m2c[0 * 4 + 3];
        temp.y = m2c[1 * 4 + 0] * v.x + m2c[1 * 4 + 1]*v.y + m2c[1 * 4 + 2]*v.z + m2c[1 * 4 + 3];
        temp.z = m2c[2 * 4 + 0] * v.x + m2c[2 * 4 + 1]*v.y + m2c[2 * 4 + 2]*v.z + m2c[2 * 4 + 3];

        //multiply K
        temp.x = K[0 * 3 + 0] * temp.x + K[0 * 3 + 1] * temp.y + K[0 * 3 + 2] * temp.z;
        temp.y = K[1 * 3 + 0] * temp.x + K[1 * 3 + 1] * temp.y + K[1 * 3 + 2] * temp.z;
        temp.z = K[2 * 3 + 0] * temp.x + K[2 * 3 + 1] * temp.y + K[2 * 3 + 2] * temp.z;

        //divide z
        temp.x /= temp.z;
        temp.y /= temp.z;
        temp.z /= temp.z;

        //denote y-axis direction is reversed
        temp.y = 480 - temp.y;

        //change to normalized axis space
        // x = 0 to 640 ~> -1 to 1
        // y = 0 to 480 ~> -1 to 1
        temp.x = (temp.x - 320) / 320;
        temp.y = (temp.y - 240) / 240;


        res.push_back(temp);
    }

    return res;
}

vector<glm::vec3> projection_parallel(vector<glm::vec3> origin, float* m2c, float* K) {
    int size = origin.size();
    vector<glm::vec3> res(size);

    //iterator is not thread safe
#pragma omp parallel for
    for (int i = 0; i < size;i++) {
        //multiply m2c
        res[i].x = m2c[0 * 4 + 0] * origin[i].x + m2c[0 * 4 + 1] * origin[i].y + m2c[0 * 4 + 2] * origin[i].z + m2c[0 * 4 + 3];
        res[i].y = m2c[1 * 4 + 0] * origin[i].x + m2c[1 * 4 + 1] * origin[i].y + m2c[1 * 4 + 2] * origin[i].z + m2c[1 * 4 + 3];
        res[i].z = m2c[2 * 4 + 0] * origin[i].x + m2c[2 * 4 + 1] * origin[i].y + m2c[2 * 4 + 2] * origin[i].z + m2c[2 * 4 + 3];

        //multiply K
        res[i].x = K[0 * 3 + 0] * res[i].x + K[0 * 3 + 1] * res[i].y + K[0 * 3 + 2] * res[i].z;
        res[i].y = K[1 * 3 + 0] * res[i].x + K[1 * 3 + 1] * res[i].y + K[1 * 3 + 2] * res[i].z;
        res[i].z = K[2 * 3 + 0] * res[i].x + K[2 * 3 + 1] * res[i].y + K[2 * 3 + 2] * res[i].z;

        //divide z
        res[i].x /= res[i].z;
        res[i].y /= res[i].z;
        res[i].z /= res[i].z;

        //denote y-axis direction is reversed
        res[i].y = 480 - res[i].y;

        //change to normalized axis space
        // x = 0 to 640 ~> -1 to 1
        // y = 0 to 480 ~> -1 to 1
        res[i].x = (res[i].x - 320) / 320;
        res[i].y = (res[i].y - 240) / 240;
    }

    return res;
}

vector<glm::vec3> cudaProjection(vector<glm::vec3> origin, float *m2c, float *k) {
    int size = origin.size() * 3;
    vector<glm::vec3> res(origin.size());
    float* deviceOrigin, *deviceTarget, *deviceM2C, *deviceK;
    cudaMalloc(&deviceOrigin,   sizeof(float) * size);
    cudaMalloc(&deviceTarget,   sizeof(float) * size);
    cudaMalloc(&deviceM2C   ,   sizeof(float) * 12);
    cudaMalloc(&deviceK     ,   sizeof(float) * 9);

    dim3 blocks(32, 1);
    dim3 grids(ceil(origin.size() / blocks.x) + 1, 1);

    auto current_time = []() {auto t = chrono::system_clock::now(); return chrono::duration_cast<chrono::nanoseconds>(t.time_since_epoch()).count(); };
    auto start = current_time();
    cudaMemcpy(deviceOrigin, &(*origin.data())[0], sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceM2C, m2c, sizeof(float) * 12, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceK, k, sizeof(float) * 9, cudaMemcpyHostToDevice);

    projection_cuda << < grids, blocks>> > (deviceOrigin, deviceTarget, size, deviceM2C, deviceK);
    cudaDeviceSynchronize();

    cudaMemcpy(&(*res.data())[0], deviceTarget, sizeof(float) * size, cudaMemcpyDeviceToHost);
    auto end = current_time();

    cout << "device projection(cuda kernel) time(ms) : " << (end - start) / 1000000. << endl;

    /*for (glm::vec3 v : res) {
        cout << v.x << v.y << v.z << endl;
    }*/

    cudaFree(deviceOrigin);
    cudaFree(deviceTarget);
    cudaFree(deviceM2C);
    cudaFree(deviceK);

    return res;
}

void drawLoop(GLFWwindow* window, GLuint programID, GLuint vertexbuffer, vector<glm::vec3> points) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glLoadIdentity();

    glUseProgram(programID);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgebuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    //glUniformMatrix4fv(MatrixID, 3, GL_FALSE, &Model[0][0]); //coordinate system
    glDrawArrays(GL_POINTS, 0, points.size());
    //glDrawElements(GL_TRIANGLES, reader.faces.size(), GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(0);

    glfwSwapBuffers(window); // 버퍼 스왑해주고
    glfwPollEvents(); // 입력 같은 여러 이벤트들 확인 후 실행
}

void drawLoop(GLFWwindow* window, GLuint programID, GLuint vertexbuffer, vector<glm::vec3> points, GLuint projID, float* m2c, GLuint cam_k_ID, float* K) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glLoadIdentity();

    glUseProgram(programID);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgebuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glUniformMatrix4fv(projID, 1, GL_FALSE, &m2c[0]); //coordinate system
    glUniformMatrix3fv(cam_k_ID, 1, GL_FALSE, &K[0]); //coordinate system
    glDrawArrays(GL_POINTS, 0, points.size());
    //glDrawElements(GL_TRIANGLES, reader.faces.size(), GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(0);

    glfwSwapBuffers(window); // 버퍼 스왑해주고
    glfwPollEvents(); // 입력 같은 여러 이벤트들 확인 후 실행
}

__global__ void projection_cuda(float* origin, float* target, int size, float* m2c, float* K) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*3;

    if (idx >= size) return;

    target[idx + 0] = origin[idx + 0] * m2c[0] + origin[idx + 1] * m2c[1] + origin[idx + 2] * m2c[2] +  m2c[3];
    target[idx + 1] = origin[idx + 0] * m2c[4] + origin[idx + 1] * m2c[5] + origin[idx + 2] * m2c[6] +  m2c[7];
    target[idx + 2] = origin[idx + 0] * m2c[8] + origin[idx + 1] * m2c[9] + origin[idx + 2] * m2c[10] + m2c[11];

    target[idx + 0] = target[idx + 0] * K[0] + target[idx + 1] * K[1] + target[idx + 2] * K[2];
    target[idx + 1] = target[idx + 0] * K[3] + target[idx + 1] * K[4] + target[idx + 2] * K[5];
    target[idx + 2] = target[idx + 0] * K[6] + target[idx + 1] * K[7] + target[idx + 2] * K[8];

    target[idx + 0] /= target[idx + 2];
    target[idx + 1] /= target[idx + 2];
    target[idx + 2] /= target[idx + 2];

    target[idx + 1] = 480 - target[idx + 1];

    target[idx + 0] = (target[idx + 0] - 320) / 320;
    target[idx + 1] = (target[idx + 1] - 240) / 240;

    return;
}