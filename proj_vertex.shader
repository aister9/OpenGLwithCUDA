#version 460 core
layout(location = 0) in vec3 vertexPosition_modelspace;
uniform mat4 m2c;
uniform mat3 K;

void main() {
	vec4 p = transpose(m2c) * vec4(vertexPosition_modelspace,1);
	vec3 res = transpose(K) * p.xyz;

	res.x /= res.z;
	res.y /= res.z;
	res.z /= res.z;

	res.y = 480 - res.y;

	res.x = (res.x - 320) / 320;
	res.y = (res.y - 240) / 240;

	gl_Position = vec4(res, 1);
}
