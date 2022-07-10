#include<iostream>
#include<fstream>
#include<sstream>
#include <ctime>
#include <emmintrin.h>
#include <immintrin.h>
using namespace std;
int count=1;
//����������Ԫ����Ŀ
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

//ѡ�����������ֱ��д���������ڲ���
class line{
public:
	int num;//��Ҫ��λ��������
    int start;//�����������һ������ֵ
    int *vector;//λ����ָ��
    line(){//�����ʼ��
        start=-1;//����ʼ��Ϊ-1����˿���ʶ�����п���
		num=(col-1)/32+1;
        vector=new int[num];///λ������ʼ��
        for(int i=0;i<num;i++)
            vector[i]=0;
    }
    bool null(){//�ж��Ƿ�Ϊ����
		return (start==-1)?1:0;
    }
    void insert(int x){//���ݶ��룬�γ�λ����
        if(start==-1)
			start=x;
        vector[x/32]|=(1<<x%32);
	}
	void gauss_avx_xor(line eliminer);
};
void line::gauss_avx_xor(line eliminer){
    int i,j;
    for(i=0;i+8<=num;i+=8){//8·�����������ȥ
        __m256i t1=_mm256_loadu_si256((__m256i*)(vector+i));
        __m256i t2=_mm256_loadu_si256((__m256i*)(eliminer.vector+i));
        t1=_mm256_xor_si256(t1,t2);
        _mm256_storeu_si256((__m256i*)vector+i,t1);//��������ұ����ڱ���Ԫ����
    }
    for(i;i<num;i++)//����ʣ��Ԫ��
         vector[i]^=eliminer.vector[i];
    for(i=num-1;i>=0;i--)//��������
        for(j=31;j>=0;j--)
            if((vector[i]&(1<<j))!=0){//ͨ������ķ�ʽ�ж��Ƿ���ڲ�Ϊ0����
                start=i*32+j;//���������Ϊ����
                return;
            }
    //����Ҳ�����Ϊ0��������Ѿ����˿���
    start=-1;
}
line *eliminer=new line[col],*eline=new line[elinenum];//������Ԫ���뱻��Ԫ��
void read(){
	//������Ԫ��
	string s;
    ifstream inf;
    inf.open("eliminer7.txt");
    while(getline(inf,s)){
        istringstream stream(s);
        int x,row=0;
        while(stream>>x){
            if(row==0)//���еĵ�һ��ֵ�����к�
				row=x;
            eliminer[row].insert(x);
        }
    }
    inf.close();

	//���뱻��Ԫ��
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
void gauss_avx(){  //��Ԫ
	int i,j;
    for(i=0;i<elinenum;i++){//�������б���Ԫ��
        while(eline[i].null()==0){//ֻҪ����Ԫ�зǿվ�һֱ�����Ԫ
            int s = eline[i].start;//�õ�����Ԫ������
            if(eliminer[s].null()==0)//�����ڶ�Ӧ��Ԫ��
                eline[i].gauss_avx_xor(eliminer[s]);//���������Ԫ
            else{
                eliminer[s]=eline[i];//������Ԫ������Ϊ��Ԫ��
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
