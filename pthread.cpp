#include<iostream>
#include<fstream>
#include<sstream>
#include<ctime>
#include<pthread.h>
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

//pthread所需参数
const int threads=8;
typedef struct{
    int t_id;
}threadparam_t;
pthread_barrier_t barrier;
pthread_barrier_t barrier2;
int num=(col-1)/32+1;//需要的位向量个数
int tmp=elinenum;

//选择定义类相较于直接写函数更易于操作
class line{
public:
    int start;//该行首项，即第一个非零值
    int *vector;//位向量指针
    line(){//矩阵初始化
        start=-1;//都初始化为-1，如此可以识别所有空行
        vector=new int[num];//位向量初始化
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
	void gauss_xor(line eliminer){
        int i,j;
		for(i=0;i<num;i++)
            vector[i]^=eliminer.vector[i];//两行异或且保存在被消元行中
        for(i=num-1;i>=0;i--)//更新首项
            for(j=31;j>=0;j--)
                if((vector[i]&(1<<j))!=0){//通过相与的方式判断是否存在不为0的项
                    start=i*32+j;//存在则更新为首项
                    return;
                }
		//如果找不到不为0的项，表明已经成了空行
        start=-1;
	}
};
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
void *threadfunc(void *param){
   threadparam_t *p=(threadparam_t *)param;
   int t_id=p->t_id;
   int i,j,k;
   for(i=col-1;i>=0;i--){//遍历消元子所有可能行号
	   if(eliminer[i].null()==0){//若该行下存在消元子
            for(j=t_id;j<elinenum;j+=threads){//遍历负责的被消元行
                if(eline[j].start==i)//若满足首行与行号一致
                    eline[j].gauss_xor(eliminer[i]);//进行消元
            }
        }
        else{//若该行号下消元子为空
            pthread_barrier_wait(&barrier);//同步
            if(t_id==0){
                for(int j=0;j<elinenum;j++){//0号线程找到匹配的被消元行
                    if(eline[j].start==i){
                        eliminer[i]=eline[j];//将被消元行升格为消元子
                        tmp=j+1;
                        break;
                     }
				}
			}
            pthread_barrier_wait(&barrier2);//同步
            int id=t_id;
            while(id<tmp)//找到该线程负责的被消元行起始位置
				id+=threads;
            for(j=id;j<elinenum;j+=threads){
                if(eline[j].start==i)
                    eline[j].gauss_xor(eliminer[i]);
            }
		}
   }
   pthread_exit(NULL);
}
void gauss_pthread(){
    pthread_barrier_init(&barrier,NULL,threads);
    pthread_barrier_init(&barrier2,NULL,threads);
    pthread_t handles[threads];
    threadparam_t param[threads];
	//创建线程
    for(int t_id=0;t_id<threads;t_id++){
        param[t_id].t_id=t_id;
        pthread_create(&handles[t_id],NULL,threadfunc,(void*)&param[t_id]);
    }
    for(int i=0;i<threads;i++)
        pthread_join(handles[i],NULL);
	//销毁线程
    pthread_barrier_destroy(&barrier);
    pthread_barrier_destroy(&barrier2);
}
int main(){
    int i=1;
    read();
    clock_t t1=clock();
    //while(i<=count){
        gauss_pthread();
        //i++;
    //}
    clock_t t2=clock();
    cout<<count<<" "<<t2-t1<<endl;
    return 0;
}
