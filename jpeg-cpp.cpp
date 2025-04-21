// Sarthak:
// DCT/IDCT math
// Quantization scaling
// Block processing loops
// Image loading and saving

// Amaan :
// Huffman tree generation
// Bitstream encoding
// Compression metrics
// Color space conversion



#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Base Quantization matrix (standard JPEG luminance)
const int BASE_Q[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 28, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

// Dynamic quantization matrix
int Q[8][8];

// Huffman Node structure
struct HuffmanNode {
    int symbol;
    float probability;
    HuffmanNode* left;
    HuffmanNode* right;
    
    HuffmanNode(int sym, float prob) : symbol(sym), probability(prob), left(nullptr), right(nullptr) {}
    
    HuffmanNode(HuffmanNode* l, HuffmanNode* r) {
        symbol = -1; // Internal node
        probability = l->probability + r->probability;
        left = l;
        right = r;
    }
    
    bool isLeaf() const {
        return left == nullptr && right == nullptr;
    }
};

// Comparison for priority queue
struct CompareNodes {
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        return a->probability > b->probability;
    }
};

// ================== DCT IMPLEMENTATION ==================
void dct2d(float block[8][8]) {
    float temp[8][8] = {0};
    const float pi = 3.14159265358979323846f;
    
    // 1D DCT on rows
    for (int y = 0; y < 8; y++) {
        for (int u = 0; u < 8; u++) {
            float sum = 0.0f;
            float cu = (u == 0) ? 1.0f/sqrt(2.0f) : 1.0f;
            
            for (int x = 0; x < 8; x++) {
                sum += block[y][x] * cos((2*x+1)*u*pi/16.0f);
            }
            
            temp[y][u] = 0.5f * cu * sum;
        }
    }
    
    // 1D DCT on columns
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float sum = 0.0f;
            float cv = (v == 0) ? 1.0f/sqrt(2.0f) : 1.0f;
            
            for (int y = 0; y < 8; y++) {
                sum += temp[y][u] * cos((2*y+1)*v*pi/16.0f);
            }
            
            block[v][u] = 0.5f * cv * sum;
        }
    }

    
}

void idct2d(float block[8][8]) {
    float temp[8][8] = {0};
    const float pi = 3.14159265358979323846f;
    
    // 1D IDCT on rows
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            float sum = 0.0f;
            
            for (int u = 0; u < 8; u++) {
                float cu = (u == 0) ? 1.0f/sqrt(2.0f) : 1.0f;
                sum += cu * block[y][u] * cos((2*x+1)*u*pi/16.0f);
            }
            
            temp[y][x] = 0.5f * sum;
        }
    }
    
    // 1D IDCT on columns
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            float sum = 0.0f;
            
            for (int v = 0; v < 8; v++) {
                float cv = (v == 0) ? 1.0f/sqrt(2.0f) : 1.0f;
                sum += cv * temp[v][x] * cos((2*y+1)*v*pi/16.0f);
            }
            
            block[y][x] = 0.5f * sum;
        }
    }
}

// Initialize quantization matrix (0.002 to 1.0)
void initQuantMatrix(float quality) {
    quality = std::max(0.002f, std::min(1.0f, quality));
    
    float scale;
    if (quality < 0.1f) {
        // Special scaling for ultra-low quality (0.002-0.1)
        scale = 50.0f * (0.1f - quality) + 5.0f;
    } else if (quality < 0.5f) {
        scale = 0.5f / quality;
    } else {
        scale = 2.0f - 2.0f * quality;
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Q[i][j] = static_cast<int>(BASE_Q[i][j] * scale);
            Q[i][j] = std::max(1, std::min(255, Q[i][j]));
        }
    }
}

// RGB to YCbCr conversion
void rgb2ycbcr(unsigned char r, unsigned char g, unsigned char b, 
               unsigned char& y, unsigned char& cb, unsigned char& cr) {
    y  = (unsigned char)( 0.299f * r + 0.587f * g + 0.114f * b);
    cb = (unsigned char)(-0.1687f * r - 0.3313f * g + 0.5f * b + 128);
    cr = (unsigned char)( 0.5f * r - 0.4187f * g - 0.0813f * b + 128);
}

// YCbCr to RGB conversion
void ycbcr2rgb(unsigned char y, unsigned char cb, unsigned char cr,
               unsigned char& r, unsigned char& g, unsigned char& b) {
    float c = y - 16.0f;
    float d = cb - 128.0f;
    float e = cr - 128.0f;
    
    r = (unsigned char)std::min(255.0f, std::max(0.0f, (1.164f * c + 1.596f * e)));
    g = (unsigned char)std::min(255.0f, std::max(0.0f, (1.164f * c - 0.392f * d - 0.813f * e)));
    b = (unsigned char)std::min(255.0f, std::max(0.0f, (1.164f * c + 2.017f * d)));
}

// Build Huffman tree
HuffmanNode* buildHuffmanTree(const std::vector<int>& symbols, const std::vector<float>& probabilities) {
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNodes> pq;
    
    for (size_t i = 0; i < symbols.size(); i++) {
        pq.push(new HuffmanNode(symbols[i], probabilities[i]));
    }
    
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();
        pq.push(new HuffmanNode(left, right));
    }
    
    return pq.top();
}

// Generate Huffman codes
void generateCodes(HuffmanNode* root, const std::string& code, std::map<int, std::string>& huffmanCodes) {
    if (root == nullptr) return;
    
    if (root->isLeaf()) {
        huffmanCodes[root->symbol] = code;
    }
    
    generateCodes(root->left, code + "0", huffmanCodes);
    generateCodes(root->right, code + "1", huffmanCodes);
}

// Calculate histogram probabilities
std::vector<float> histcounts(const std::vector<int>& values, int num_bins) {
    std::vector<float> counts(num_bins, 0.0f);
    
    for (int val : values) {
        if (val >= 0 && val < num_bins) {
            counts[val]++;
        }
    }
    
    if (!values.empty()) {
        for (float& count : counts) {
            count /= values.size();
        }
    }
    
    return counts;
}

// Main compression function
void compressJPEG(const std::string& input_file, const std::string& output_file, float quality) {
    // Initialize quantization matrix
    initQuantMatrix(quality);
    
    // Load image
    int width, height, channels;
    unsigned char* img_data = stbi_load(input_file.c_str(), &width, &height, &channels, 3);
    if (!img_data) {
        std::cerr << "Error loading image: " << input_file << std::endl;
        return;
    }

    // Convert to YCbCr
    std::vector<unsigned char> ycbcr_data(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb2ycbcr(img_data[3*i], img_data[3*i+1], img_data[3*i+2],
                 ycbcr_data[3*i], ycbcr_data[3*i+1], ycbcr_data[3*i+2]);
        
        // Simple 4:2:0 subsampling
        if (i % 2 != 0 || (i/width) % 2 != 0) {
            int base_idx = ((i/width)/2 * 2) * width + (i%width)/2 * 2;
            ycbcr_data[3*i+1] = ycbcr_data[3*base_idx+1]; // Cb
            ycbcr_data[3*i+2] = ycbcr_data[3*base_idx+2]; // Cr
        }
    }

    // Process each 8x8 block
    std::vector<unsigned char> compressed_data(width * height * 3);
    std::vector<int> quantized_coeffs;
    
    for (int channel = 0; channel < 3; channel++) {
        for (int y = 0; y <= height - 8; y += 8) {
            for (int x = 0; x <= width - 8; x += 8) {
                float block[8][8];
                
                // Extract block
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = ((y+j)*width + (x+i)) * 3 + channel;
                        block[j][i] = (float)ycbcr_data[idx] - 128.0f;
                    }
                }
                
                // Forward DCT
                dct2d(block);
                
                // Quantization
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) {
                        block[j][i] = round(block[j][i] / Q[j][i]);
                        quantized_coeffs.push_back((int)block[j][i]);
                        block[j][i] *= Q[j][i];
                    }
                }
                
                // Inverse DCT
                idct2d(block);
                
                // Store reconstructed block
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = ((y+j)*width + (x+i)) * 3 + channel;
                        float val = block[j][i] + 128.0f;
                        compressed_data[idx] = (unsigned char)std::min(255.0f, std::max(0.0f, val));
                    }
                }
            }
        }
    }
    
    // Huffman coding
    std::vector<int> symbols;
    {
        std::set<int> unique_set(quantized_coeffs.begin(), quantized_coeffs.end());
        symbols.assign(unique_set.begin(), unique_set.end());
    }
    
    std::vector<float> prob = histcounts(quantized_coeffs, 
                                       *std::max_element(symbols.begin(), symbols.end()) + 1);
    
    std::vector<int> non_zero_symbols;
    std::vector<float> non_zero_prob;
    for (size_t i = 0; i < symbols.size(); i++) {
        int symbol = symbols[i];
        if (symbol < prob.size() && prob[symbol] > 0) {
            non_zero_symbols.push_back(symbol);
            non_zero_prob.push_back(prob[symbol]);
        }
    }
    
    HuffmanNode* huffmanTree = buildHuffmanTree(non_zero_symbols, non_zero_prob);
    std::map<int, std::string> huffmanCodes;
    generateCodes(huffmanTree, "", huffmanCodes);
    
    // Calculate compression stats
    size_t original_size = width * height * channels * 8;
    size_t compressed_size = 0;
    for (int val : quantized_coeffs) {
        compressed_size += huffmanCodes[val].length();
    }
    
    float achieved_ratio = (float)original_size / compressed_size;
    std::cout << "Requested quality: " << quality << "\n";
    std::cout << "Achieved compression ratio: " << achieved_ratio << ":1\n";
    
    // Convert back to RGB and save
    std::vector<unsigned char> output_rgb(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        ycbcr2rgb(compressed_data[3*i], compressed_data[3*i+1], compressed_data[3*i+2],
                  output_rgb[3*i], output_rgb[3*i+1], output_rgb[3*i+2]);
    }
    
    stbi_write_jpg(output_file.c_str(), width, height, 3, output_rgb.data(), 90);
    std::cout << "Compressed image saved to: " << output_file << "\n";
    
    stbi_image_free(img_data);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <input.jpg> <output.jpg> <quality(0.002-1.0)>\n";
        std::cout << "Quality guide:\n";
        std::cout << "  1.0 = Best quality (low compression)\n";
        std::cout << "  0.5 = Balanced quality\n";
        std::cout << "  0.1 = High compression\n";
        std::cout << "  0.002 = Extreme compression (visible artifacts)\n";
        return 1;
    }

    float quality;
    std::istringstream iss(argv[3]);
    if (!(iss >> quality) || quality < 0.002f || quality > 1.0f) {
        std::cerr << "Error: Quality must be between 0.002 and 1.0\n";
        return 1;
    }

    compressJPEG(argv[1], argv[2], quality);
    return 0;
}