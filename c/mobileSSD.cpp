/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "mobileSSD.h"
#include "imageNet.cuh"

#include "cudaMappedMemory.h"
#include "cudaFont.h"

#include "commandLine.h"
#include "filesystem.h"


#define OUTPUT_CVG  0	// Caffe has output coverage (confidence) heat map
#define OUTPUT_BBOX 1	// Caffe has separate output layer for bounding box

#define OUTPUT_UFF  0	// UFF has primary output containing detection results
#define OUTPUT_NUM	1	// UFF has secondary output containing one detection count

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"
//#define DEBUG_CLUSTERING


// constructor
mobileSSD::mobileSSD( float meanPixel ) : tensorNet()
{
	mCoverageThreshold = mobileSSD_DEFAULT_THRESHOLD;
	mMeanPixel         = meanPixel;
	mCustomClasses     = 0;
	mNumClasses        = 0;

	mClassColors[0]   = NULL; // cpu ptr
	mClassColors[1]   = NULL; // gpu ptr
	
	mDetectionSets[0] = NULL; // cpu ptr
	mDetectionSets[1] = NULL; // gpu ptr
	mDetectionSet     = 0;
	mMaxDetections    = 0;
}


// destructor
mobileSSD::~mobileSSD()
{
	if( mDetectionSets != NULL )
	{
		CUDA(cudaFreeHost(mDetectionSets[0]));
		
		mDetectionSets[0] = NULL;
		mDetectionSets[1] = NULL;
	}
	
	if( mClassColors != NULL )
	{
		CUDA(cudaFreeHost(mClassColors[0]));
		
		mClassColors[0] = NULL;
		mClassColors[1] = NULL;
	}
}


// init
bool mobileSSD::init( const char* prototxt, const char* model, const char* mean_binary, const char* class_labels, 
			 	  float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
				  uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	printf("\n");
	printf("mobileSSD -- loading detection network model from:\n");
	printf("          -- prototxt     %s\n", CHECK_NULL_STR(prototxt));
	printf("          -- model        %s\n", CHECK_NULL_STR(model));
	printf("          -- input_blob   '%s'\n", CHECK_NULL_STR(input_blob));
	printf("          -- output_cvg   '%s'\n", CHECK_NULL_STR(coverage_blob));
	printf("          -- output_bbox  '%s'\n", CHECK_NULL_STR(bbox_blob));
	printf("          -- mean_pixel   %f\n", mMeanPixel);
	printf("          -- mean_binary  %s\n", CHECK_NULL_STR(mean_binary));
	printf("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);

	//net->EnableDebug();
	
	// create list of output names	
	std::vector<std::string> output_blobs;

	if( coverage_blob != NULL )
		output_blobs.push_back(coverage_blob);

	if( bbox_blob != NULL )
		output_blobs.push_back(bbox_blob);
	
	// load the model
	if( !LoadNetwork(prototxt, model, mean_binary, input_blob, output_blobs, 
				  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("mobileSSD -- failed to initialize.\n");
		return false;
	}
	
	// allocate detection sets
	if( !allocDetections() )
		return false;

	// load class descriptions
	loadClassDesc(class_labels);
	defaultClassDesc();
	
	// set default class colors
	if( !defaultColors() )
		return false;

	// set the specified threshold
	SetThreshold(threshold);

	return true;
}


// Create
mobileSSD* mobileSSD::Create( const char* prototxt, const char* model, float mean_pixel, const char* class_labels,
						float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	mobileSSD* net = new mobileSSD(mean_pixel);
	
	if( !net )
		return NULL;

	if( !net->init(prototxt, model, NULL, class_labels, threshold, input_blob, coverage_blob, bbox_blob,
				maxBatchSize, precision, device, allowGPUFallback) )
		return NULL;

	return net;
}


// Create
mobileSSD* mobileSSD::Create( const char* prototxt, const char* model, const char* mean_binary, const char* class_labels, 
						float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	mobileSSD* net = new mobileSSD();
	
	if( !net )
		return NULL;

	if( !net->init(prototxt, model, mean_binary, class_labels, threshold, input_blob, coverage_blob, bbox_blob,
				maxBatchSize, precision, device, allowGPUFallback) )
		return NULL;
	
	return net;
}


// Create
mobileSSD* mobileSSD::Create( int argc, char** argv )
{
	mobileSSD* net = NULL;

	// parse command line parameters
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "ssd-mobilenet-v2");

	float threshold = cmdLine.GetFloat("threshold");
	
	if( threshold == 0.0f )
		threshold = mobileSSD_DEFAULT_THRESHOLD;
	
	int maxBatchSize = cmdLine.GetInt("batch_size");
	
	if( maxBatchSize < 1 )
		maxBatchSize = DEFAULT_MAX_BATCH_SIZE;


	const char* prototxt     = cmdLine.GetString("prototxt");
	const char* input        = cmdLine.GetString("input_blob");
	const char* out_blob     = cmdLine.GetString("output_blob");
	const char* out_cvg      = cmdLine.GetString("output_cvg");
	const char* out_bbox     = cmdLine.GetString("output_bbox");
	const char* class_labels = cmdLine.GetString("class_labels");

	if( !input ) 	
		input = mobileSSD_DEFAULT_INPUT;

	if( !out_blob )
	{
		if( !out_cvg )  out_cvg  = mobileSSD_DEFAULT_COVERAGE;
		if( !out_bbox ) out_bbox = mobileSSD_DEFAULT_BBOX;
	}

	float meanPixel = cmdLine.GetFloat("mean_pixel");

	net = mobileSSD::Create(prototxt, modelName, meanPixel, class_labels, threshold, input, 
						out_blob ? NULL : out_cvg, out_blob ? out_blob : out_bbox, maxBatchSize);

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// set overlay alpha value
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", mobileSSD_DEFAULT_ALPHA));

	return net;
}
	

// allocDetections
bool mobileSSD::allocDetections()
{

	printf("W = %u  H = %u  C = %u\n", DIMS_W(mOutputs[OUTPUT_UFF].dims), DIMS_H(mOutputs[OUTPUT_UFF].dims), DIMS_C(mOutputs[OUTPUT_UFF].dims));
	mMaxDetections = DIMS_H(mOutputs[OUTPUT_UFF].dims) * DIMS_C(mOutputs[OUTPUT_UFF].dims);
	mNumClasses = mClassDesc.size();
	
	printf("mobileSSD -- maximum bounding boxes:  %u\n", mMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	if( !cudaAllocMapped((void**)&mDetectionSets[0], (void**)&mDetectionSets[1], det_size) )
		return false;
	
	memset(mDetectionSets[0], 0, det_size);
	return true;
}

	
// defaultColors
bool mobileSSD::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();
	if( !cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], numClasses * sizeof(float4)) )
		return false;

	// if there are a large number of classes (MS COCO)
	// programatically generate the class color map
	if( numClasses > 10 )
	{
		// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
		#define bitget(byteval, idx)	((byteval & (1 << idx)) != 0)

		for( int i=0; i < numClasses; i++ )
		{
			int r = 0;
			int g = 0;
			int b = 0;
			int c = i;

			for( int j=0; j < 8; j++ )
			{
				r = r | (bitget(c, 0) << 7 - j);
				g = g | (bitget(c, 1) << 7 - j);
				b = b | (bitget(c, 2) << 7 - j);
				c = c >> 3;
			}

			mClassColors[0][i*4+0] = r;
			mClassColors[0][i*4+1] = g;
			mClassColors[0][i*4+2] = b;
			mClassColors[0][i*4+3] = mobileSSD_DEFAULT_ALPHA; 

			//printf(LOG_TRT "color %02i  %3i %3i %3i %3i\n", i, (int)r, (int)g, (int)b, (int)alpha);
		}
	}
	else
	{
		// blue colors, except class 1 is green
		for( uint32_t n=0; n < numClasses; n++ )
		{
			if( n != 1 )
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 200.0f;	// g
				mClassColors[0][n*4+2] = 255.0f;	// b
				mClassColors[0][n*4+3] = mobileSSD_DEFAULT_ALPHA;	// a
			}
			else
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 255.0f;	// g
				mClassColors[0][n*4+2] = 175.0f;	// b
				mClassColors[0][n*4+3] = 75.0f;	// a
			}
		}
	}

	return true;
}


// defaultClassDesc
void mobileSSD::defaultClassDesc()
{
	const uint32_t numClasses = GetNumClasses();
	const int syn = 9;  // length of synset prefix (in characters)
	
	// assign defaults to classes that have no info
	for( uint32_t n=mClassDesc.size(); n < numClasses; n++ )
	{
		char syn_str[10];
		sprintf(syn_str, "n%08u", mCustomClasses);

		char desc_str[16];
		sprintf(desc_str, "class #%u", mCustomClasses);

		mClassSynset.push_back(syn_str);
		mClassDesc.push_back(desc_str);

		mCustomClasses++;
	}
}


// loadClassDesc
bool mobileSSD::loadClassDesc( const char* filename )
{
	//printf("mobileSSD -- model has %u object classes\n", GetNumClasses());

	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		printf("mobileSSD -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("mobileSSD -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class descriptions
	char str[512];

	while( fgets(str, 512, f) != NULL )
	{
		const int syn = 9;  // length of synset prefix (in characters)
		const int len = strlen(str);
		
		if( len > syn && str[0] == 'n' && str[syn] == ' ' )
		{
			str[syn]   = 0;
			str[len-1] = 0;
	
			const std::string a = str;
			const std::string b = (str + syn + 1);
	
			//printf("a=%s b=%s\n", a.c_str(), b.c_str());

			mClassSynset.push_back(a);
			mClassDesc.push_back(b);
		}
		else if( len > 0 )	// no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", mCustomClasses);

			//printf("a=%s b=%s (custom non-synset)\n", a, str);
			mCustomClasses++;

			if( str[len-1] == '\n' )
				str[len-1] = 0;

			mClassSynset.push_back(a);
			mClassDesc.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("mobileSSD -- loaded %zu class info entries\n", mClassDesc.size());
	
	//for( size_t n=0; n < mClassDesc.size(); n++ )
		//printf("          -- %s '%s'\n", mClassSynset[n].c_str(), mClassDesc[n].c_str());

	if( mClassDesc.size() == 0 )
		return false;

	mNumClasses = mClassDesc.size();

	printf("mobileSSD -- number of object classes:  %u\n", mNumClasses);
	mClassPath = path;	
	return true;
}


#if 0
inline static bool rectOverlap(const float6& r1, const float6& r2)
{
    return ! ( r2.x > r1.z  
        || r2.z < r1.x
        || r2.y > r1.w
        || r2.w < r1.y
        );
}
#endif


// Detect
int mobileSSD::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	Detection* det = mDetectionSets[0] + mDetectionSet * GetMaxDetections();

	if( detections != NULL )
		*detections = det;

	mDetectionSet++;

	if( mDetectionSet >= mNumDetectionSets )
		mDetectionSet = 0;
	
	return Detect(input, width, height, det, overlay);
}

	
// Detect
int mobileSSD::Detect( float* rgba, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay )
{
	if( !rgba || width == 0 || height == 0 || !detections )
	{
		printf(LOG_TRT "mobileSSD::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);


	if( mMeanPixel != 0.0f )
	{
		if( CUDA_FAILED(cudaPreImageNetMeanBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										make_float3(mMeanPixel, mMeanPixel, mMeanPixel), GetStream())) )
		{
			printf(LOG_TRT "mobileSSD::Detect() -- cudaPreImageNetMean() failed\n");
			return -1;
		}
	}
	else
	{
		if( CUDA_FAILED(cudaPreImageNetBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, GetStream())) )
		{
			printf(LOG_TRT "mobileSSD::Detect() -- cudaPreImageNet() failed\n");
			return -1;
		}
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	// process with TensorRT
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA, mOutputs[1].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "mobileSSD::Detect() -- failed to execute TensorRT context\n");
		return -1;
	}


	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// post-processing / clustering
	int numDetections = 0;

	const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
	const int rawParameters = 7;  // seems a bug
	printf("rawParameters %i\n", rawParameters);

	#ifdef DEBUG_CLUSTERING	
			printf(LOG_TRT "detectNet::Detect() -- %i unfiltered detections\n", rawDetections);
	#endif


	// filter the raw detections by thresholding the confidence
	for( int n=0; n < rawDetections; n++ )
	{
		float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

		if( object_data[2] < mCoverageThreshold )
			continue;

		detections[numDetections].Instance   = numDetections; //(uint32_t)object_data[0];
		// [Liang Ru] be care this field, needs to figure out the class_id while multiple classes
		detections[numDetections].ClassID    = (uint32_t)object_data[1] - 1; 
		detections[numDetections].Confidence = object_data[2];
		detections[numDetections].Left       = object_data[3] * width;
		detections[numDetections].Top        = object_data[4] * height;
		detections[numDetections].Right      = object_data[5] * width;
		detections[numDetections].Bottom	  = object_data[6] * height;

		if( detections[numDetections].ClassID >= mNumClasses )
		{
			printf(LOG_TRT "detectNet::Detect() -- detected object has invalid classID (%u)\n", detections[numDetections].ClassID);
			detections[numDetections].ClassID = 0;
		}

		if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
			continue;

		numDetections += clusterDetections(detections, numDetections);
	}

	// sort the detections by confidence value
	sortDetections(detections, numDetections);


	PROFILER_END(PROFILER_POSTPROCESS);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(rgba, rgba, width, height, detections, numDetections, overlay) )
			printf(LOG_TRT "mobileSSD::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return numDetections;
}


// clusterDetections
int mobileSSD::clusterDetections( Detection* detections, int n, float threshold )
{
	if( n == 0 )
		return 1;

	// test each detection to see if it intersects
	for( int m=0; m < n; m++ )
	{
		if( detections[n].Intersects(detections[m], threshold) )	// TODO NMS or different threshold for same classes?
		{
			// if the intersecting detections have different classes, pick the one with highest confidence
			// otherwise if they have the same object class, expand the detection bounding box
			if( detections[n].ClassID != detections[m].ClassID )
			{
				if( detections[n].Confidence > detections[m].Confidence )
				{
					detections[m] = detections[n];

					detections[m].Instance = m;
					detections[m].ClassID = detections[n].ClassID;
					detections[m].Confidence = detections[n].Confidence;					
				}
			}
			else
			{
				detections[m].Expand(detections[n]);
				detections[m].Confidence = fmaxf(detections[n].Confidence, detections[m].Confidence);
			}

			return 0; // merged detection
		}
	}

	return 1;	// new detection
}


// sortDetections
void mobileSSD::sortDetections( Detection* detections, int numDetections )
{
	if( numDetections < 2 )
		return;

	// order by area (descending) or confidence (ascending)
	for( int i=0; i < numDetections-1; i++ )
	{
		for( int j=0; j < numDetections-i-1; j++ )
		{
			if( detections[j].Area() < detections[j+1].Area() ) //if( detections[j].Confidence > detections[j+1].Confidence )
			{
				const Detection det = detections[j];
				detections[j] = detections[j+1];
				detections[j+1] = det;
			}
		}
	}

	// renumber the instance ID's
	for( int i=0; i < numDetections; i++ )
		detections[i].Instance = i;	
}


// from mobileSSD.cu
cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, mobileSSD::Detection* detections, int numDetections, float4* colors );

// Overlay
bool mobileSSD::Overlay( float* input, float* output, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections, uint32_t flags )
{
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	if( flags == 0 )
	{
		printf(LOG_TRT "mobileSSD -- Overlay() was called with OVERLAY_NONE, returning false\n");
		return false;
	}

	// bounding box overlay
	if( flags & OVERLAY_BOX )
	{
		if( CUDA_FAILED(cudaDetectionOverlay((float4*)input, (float4*)output, width, height, detections, numDetections, (float4*)mClassColors[1])) )
			return false;
	}

	// class label overlay
	if( (flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) )
	{
		static cudaFont* font = NULL;

		// make sure the font object is created
		if( !font )
		{
			font = cudaFont::Create(adaptFontSize(width));
	
			if( !font )
			{
				printf(LOG_TRT "mobileSSD -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
		std::vector< std::pair< std::string, int2 > > labels;

		for( uint32_t n=0; n < numDetections; n++ )
		{
			const char* className  = GetClassDesc(detections[n].ClassID);
			const float confidence = detections[n].Confidence * 100.0f;
			const int2  position   = make_int2(detections[n].Left+5, detections[n].Top+3);
			
			if( flags & OVERLAY_CONFIDENCE )
			{
				char str[256];

				if( (flags & OVERLAY_LABEL) && (flags & OVERLAY_CONFIDENCE) )
					sprintf(str, "%s %.1f%%", className, confidence);
				else
					sprintf(str, "%.1f%%", confidence);

				labels.push_back(std::pair<std::string, int2>(str, position));
			}
			else
			{
				// overlay label only
				labels.push_back(std::pair<std::string, int2>(className, position));
			}
		}

		font->OverlayText((float4*)input, width, height, labels, make_float4(255,255,255,255));
	}
	
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// OverlayFlagsFromStr
uint32_t mobileSSD::OverlayFlagsFromStr( const char* str_user )
{
	if( !str_user )
		return OVERLAY_BOX;

	// copy the input string into a temporary array,
	// because strok modifies the string
	const size_t str_length = strlen(str_user);

	if( str_length == 0 )
		return OVERLAY_BOX;

	char* str = (char*)malloc(str_length + 1);

	if( !str )
		return OVERLAY_BOX;

	strcpy(str, str_user);

	// tokenize string by delimiters ',' and '|'
	const char* delimiters = ",|";
	char* token = strtok(str, delimiters);

	if( !token )
	{
		free(str);
		return OVERLAY_BOX;
	}

	// look for the tokens:  "box", "label", and "none"
	uint32_t flags = OVERLAY_NONE;

	while( token != NULL )
	{
		//printf("%s\n", token);

		if( strcasecmp(token, "box") == 0 )
			flags |= OVERLAY_BOX;
		else if( strcasecmp(token, "label") == 0 || strcasecmp(token, "labels") == 0 )
			flags |= OVERLAY_LABEL;
		else if( strcasecmp(token, "conf") == 0 || strcasecmp(token, "confidence") == 0 )
			flags |= OVERLAY_CONFIDENCE;

		token = strtok(NULL, delimiters);
	}	

	free(str);
	return flags;
}


// SetClassColor
void mobileSSD::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors[0] )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[0][i+0] = r;
	mClassColors[0][i+1] = g;
	mClassColors[0][i+2] = b;
	mClassColors[0][i+3] = a;
}


// SetOverlayAlpha
void mobileSSD::SetOverlayAlpha( float alpha )
{
	const uint32_t numClasses = GetNumClasses();

	for( uint32_t n=0; n < numClasses; n++ )
		mClassColors[0][n*4+3] = alpha;
}
