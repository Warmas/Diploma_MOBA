#version 330
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aOffset;

uniform mat4 view;
uniform mat4 projection;
void main(){
    gl_Position = projection * view * vec4(aPos + aOffset, 0.0, 1.0);
}