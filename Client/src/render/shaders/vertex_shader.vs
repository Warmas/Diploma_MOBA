#version 330
layout(location = 0) in vec2 aPos;

uniform vec2 offset;
uniform mat4 view;
uniform mat4 projection;
void main(){
    gl_Position = projection * view * vec4(aPos + offset, 0.0, 1.0);
}