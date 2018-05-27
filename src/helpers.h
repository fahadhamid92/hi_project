#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>
#include <stdio.h>


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define MAX_WORD_LENGTH 100
#define NUM_ALPHABET 26



#define NUM_CHILDREN 8


typedef struct node_t {
    char * word;
    struct node_t * next;
    struct node_t ** children;
} Node;


Node * makeNode();



// Inserts the given word to trie, modifying the structure if necessary
void insert(Node * root, char * word);

// Ruilds trie with words in the specified file.
// prints error message if it's not a valid file.
Node * build(FILE * fp);

// Returns a node that contains dictionary word which corresponds
// to the given key sequence. Returns NULL if the keySeq is invalid
Node * search(char * keySeq, Node * root);

// Returns 0-7 child index that corresponds with the given letter
int getChildIndex(char letter);

// Clears/frees the given node and its children.
void clear(Node * node);

// Frees the linked list of node with the given front
void freeList(Node * front);

// Frees a node referenced by the given node pointer
void freeNode(Node * node);

// Prints the given trie in preorder
void printTrie(Node * root, int level);

// Prints the numTabs many tabs
void printTabs(int numTabs);




// Allocated 'size' memory to heap and verifies the result.
// If malloc fails, then program exits with error message.
void * mallocVerif(size_t size);




bool rectInImage(cv::Rect rect, cv::Mat image);
bool inMat(cv::Point p,int rows,int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);

#endif