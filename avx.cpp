#include<iostream>
#include<fstream>
#include<sstream>
#include <ctime>
#include <emmintrin.h>
#include <immintrin.h>
using namespace std;
int count=1;
//列数，被消元行数目
//int col=130,elinenum=8;
//int col=254,elinenum=53;
//int col=562,elinenum=53;
//int col=1011,elinenum=263;
//int col=2362,elinenum=453;
//int col=3799,elinenum=1953;
int col=8399,elinenum=4535;
//int col=23045,elinenum=14325;
//int col=37960,elinenum=14921;
//int col=43577,elinenum=54274;
//int col=85401,elinenum=756;

//选择定义类相较于直接写函数更易于操作
class line{
public:
	int num;//需要的位向量个数
    int start;//该行首项，即第一个非零值
    int *vector;//位向量指针
    line(){//矩阵初始化
        start=-1;//都初始化为-1，如此可以识别所有空行
		num=(col-1)/32+1;
        vector=new int[num];///位向量初始化
        for(int i=0;i<num;i++)
            vector[i]=0;
    }
    bool null(){//判断是否为空行
		return (start==-1)?1:0;
    }
    void insert(int x){//数据读入，形成位向量
        if(start==-1)
			start=x;
        vector[x/32]|=(1<<x%32);
	}
	void gauss_avx_xor(line eliminer);
};
void line::gauss_avx_xor(line eliminer){
    int i,j;
    for(i=0;i+8<=num;i+=8){//8路向量化异或消去
        __m256i t1=_mm256_loadu_si256((__m256i*)(vector+i));
        __m256i t2=_mm256_loadu_si256((__m256i*)(eliminer.vector+i));
        t1=_mm256_xor_si256(t1,t2);
        _mm256_storeu_si256((__m256i*)vector+i,t1);//两行异或且保存在被消元行中
    }
    for(i;i<num;i++)//处理剩余元素
         vector[i]^=eliminer.vector[i];
    for(i=num-1;i>=0;i--)//更新首项
        for(j=31;j>=0;j--)
            if((vector[i]&(1<<j))!=0){//通过相与的方式判断是否存在不为0的项
                start=i*32+j;//存在则更新为首项
                return;
            }
    //如果找不到不为0的项，表明已经成了空行
    start=-1;
}
line *eliminer=new line[col],*eline=new line[elinenum];//定义消元子与被消元行
void read(){
	//读入消元子
	string s;
    ifstream inf;
    inf.open("eliminer7.txt");
    while(getline(inf,s)){
        istringstream stream(s);
        int x,row=0;
        while(stream>>x){
            if(row==0)//新行的第一个值就是行号
				row=x;
            eliminer[row].insert(x);
        }
    }
    inf.close();

	//读入被消元行
    ifstream inf2;
    inf2.open("eline7.txt");
    int row=0;
    while(getline(inf2,s)){
        istringstream stream(s);
        int x;
        while(stream>>x)
            eline[row].insert(x);
        row++;
    }
    inf2.close();
}
void gauss_avx(){  //消元
	int i,j;
    for(i=0;i<elinenum;i++){//遍历所有被消元行
        while(eline[i].null()==0){//只要被消元行非空就一直异或消元
            int s = eline[i].start;//得到被消元行首项
            if(eliminer[s].null()==0)//若存在对应消元子
                eline[i].gauss_avx_xor(eliminer[s]);//进行异或消元
            else{
                eliminer[s]=eline[i];//否则被消元行升格为消元子
                break;
            }
        }
    }
}
int main(){
    int i=1;
    read();
    clock_t t1=clock();
    //while(i<=count){
        gauss_avx();
        //i++;
    //}
    clock_t t2=clock();
    cout<<count<<" "<<t2-t1<<endl;
    return 0;
}
