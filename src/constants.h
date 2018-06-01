#ifndef CONSTANTS_H
#define CONSTANTS_H

// Debugging
const bool kPlotVectorField = false;

const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.005;

const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;

const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;

const bool kEnableEyeCorner = false;

#endif