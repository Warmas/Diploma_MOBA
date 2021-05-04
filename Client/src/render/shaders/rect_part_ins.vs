#version 330
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aMoving;
layout(location = 2) in vec2 aOffset;
layout(location = 3) in float aPercentage;

uniform mat4 projection;
uniform float width;
void main(){
    gl_Position = projection * vec4(aPos.x + aOffset.x - (aMoving * width * aPercentage), aPos.y + aOffset.y, 0.0, 1.0);
}