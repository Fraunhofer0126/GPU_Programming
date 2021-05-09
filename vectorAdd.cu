#include<iostream>
#include<cmath>
using namespace std;
const int MAX = 100;

template<class T>
__global__
void vecAddKernel(T *A, T *B, T *C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n) C[i] = A[i]+B[i];
}
template<class T>
void vecAdd(T *A, T *B, T *C, int n){
    int size = n * sizeof(T);
    T *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_C, size);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
}
int main(){
    int n; cin>>n;
    int A[MAX], B[MAX], C[MAX];
    for(int i = 0; i < n; i++)
        cin >> A[i];
    for(int i = 0; i < n; i++)
        cin >> B[i];
    vecAdd<int>(A, B, C, n);
    for(int i = 0; i < n; i++)
        cout<<C[i]<<' ';
    cout<<endl;
    return 0;
}
