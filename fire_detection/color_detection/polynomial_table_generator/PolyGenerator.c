#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#define DIM 3
#define SIZE 225

long double fu(int val)
{
     long double result=0;
     result-=(2.6*pow(10,-10)*pow(val,7));
     result+=(3.3*pow(10,-7)*pow(val,6));
     result-=(1.7*pow(10,-4)*pow(val,5));
     result+=(5.16*pow(10,-2)*pow(val,4));
     result-=(9.1*pow(val,3));
     result+=(9.6*pow(10,2)*pow(val,2));
     result-=(5.6*pow(10,4)*val);
     result+=(1.4*pow(10,6));
     return (result/100);
}

long double fl(int val)
{
    long double result=0;
    result-=(6.77*pow(10,-8)*pow(val,5));
    result+=(5.5*pow(10,-5)*pow(val,4));
    result-=(1.76*pow(10,-2)*pow(val,3));
    result+=(2.78*pow(val,2));
    result-=(2.15*pow(10,2)*val);
    result+=(6.62*pow(10,3));
    return result;
}

long double fd(int val)
{
    long double result=0;
    result+=(1.81*pow(10,-4)*pow(val,4));
    result-=(1.02*pow(10,-1)*pow(val,3));
    result+=(2.17*10*pow(val,2));
    result-=(2.05*pow(10,3)*val);
    result+=(7.29*pow(10,4));

    return (result/10);
}

int main()
{
    system("clear");

    FILE* file=fopen("Poly_Lookup_Table.txt","w+");

    long double lookup_table[SIZE][DIM];

    /***
    lookup_table[0] -> fu()
    lookup_table[1] -> fl()
    lookup_table[2] -> fd()
    ***/

    for(int i=16;i<=240;i++)
    {
        lookup_table[i-16][0]=(fu(i));
        lookup_table[i-16][1]=(fl(i));
        lookup_table[i-16][2]=(fd(i));
    }

    for(int i=0;i<=240-16;i++)
    {
        fprintf(file,"%Lf,%Lf,%Lf\n",lookup_table[i][0],lookup_table[i][1],lookup_table[i][2]);
        printf("fu(%d)=%Lf, fl(%d)=%Lf, fd(%d)=%Lf\n",i+16,lookup_table[i][0],i+16,lookup_table[i][1],i+16,lookup_table[i][2]);
    }
    
    return 0;
}