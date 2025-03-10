//
//    TransTessellate2D : A general Trans-dimensional Tessellation program
//    for 2D Cartesian problems.
//
//    Copyright (C) 2014 - 2019 Rhys Hawkins
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//

#pragma once
#ifndef pathutil_hpp
#define pathutil_hpp

void mkpath(const char *prefix, const char *filename, char *path);

void mkmodelpath(int mi, const char *prefix, const char *filename, char *path);

void mkrankpath(int rank, const char *prefix, const char *filename, char *path);

void mkmodelrankpath(int mi, int rank, const char *prefix, const char *filename, char *path);

#endif // pathutil_hpp
