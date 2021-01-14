#pragma once

#include <iostream>

// const unsigned gridSize = 64;
const char edge = '+';
const char horizontalWall = '-';
const char verticalWall = '|';
const float min = 273.0f;
const float max = 333.0f;
const std::string tempString = " .,:;+*#";

void printWall(std::ostream& out, unsigned xGridsize){
    out << edge;
    for(unsigned i = 0; i < xGridsize; i++){
        out << horizontalWall;
    }
    out << edge << '\n';
}

void printValue(float val, std::ostream& out){
    out << tempString[static_cast<int>((val - min) / (max - min) * (tempString.size()-1))];
}

template<class T>
void printLine(T buffer, unsigned xSize, std::ostream& out, unsigned xGridsize, unsigned line, unsigned lineWidth){
    out << verticalWall;
    unsigned elementsPerPixel = xSize / xGridsize;
    for(unsigned i = 0; i < xGridsize; i++){
        float val = 0.0f;
        for(unsigned y = 0; y < lineWidth; y++){
            for(unsigned x = 0; x < elementsPerPixel; x++){
                val += buffer[((line * lineWidth) + y) * xSize + i * elementsPerPixel + x];
            }
        }
        val /= (lineWidth * elementsPerPixel);
        printValue(val, out);
    }
    out << verticalWall << '\n';
}

template<class T>
void printStencil(T buffer, unsigned xSize, unsigned ySize, std::ostream& out, unsigned xGridsize = 32, unsigned yGridsize = 16){
    assert(xSize % xGridsize == 0 && "Outputsize not devidable");
    assert(ySize % yGridsize == 0 && "Outputsize not devidable");

    printWall(out, xGridsize);
    for(unsigned i = 0; i < yGridsize; i++){
        printLine(buffer, xSize, out, xGridsize, i, ySize / yGridsize);
    }
    printWall(out, xGridsize);
    out << std::endl;
}