#include <bits/stdc++.h>
using namespace std;
#define sz(x) ((int)size(x))
#define sig(x) (x<0?-1:1)
#define all(x) begin(x),end(x)
using ll=long long;
template <typename T> using vec=vector<T>;
template <typename T> struct vec2:vector<vector<T>> {vec2()=default;vec2(int n, int m, T val=T()):vector<vector<T>>(n,vector<T>(m,val)){}};
template <typename T> void vprint(T st, T nd) {auto it=st; while(next(it)!=nd){cout<<*it<<' '; it=next(it);}cout<<*it<<'\n';}

// # training examples
const int m=3807/2;
// # features
const int nf=14;
// # degree
const int nd=2;

// learning rate
const double a=1e-3;
// iters
const int iters=1e4;

void input(vec<vec<double>> &tx, vec<double> &ty, vec<vec<double>> &testx, vec<double> &testy) {
    ifstream ifs("data/numerical_movie.csv");
    string buf; getline(ifs,buf); // skip first line

    for (int i=0; i<2*m; ++i) {
        double tmp; char c; ifs>>tmp>>c;
        bool testset=i>=m;
        vec<double> xi;
        for (int j=0; j<nf; ++j) {
            double x; ifs>>x;
            xi.push_back(x);

            // comma
            ifs>>c;
        }
        (testset?testx:tx).push_back(xi);

        double y; ifs>>y;
        (testset?testy:ty).push_back(y);
    }
}

// modifies x, mu, sd
// normalize by feature
// x: m*nf
// mu: nf
// sd: nf
void feature_scale(vec<vec<double>> &x, vec<double> &mu, vec<double> &sd) {
    for (int j=0; j<nf; ++j) {
        // mean
        for (int i=0; i<m; ++i) {
            mu[j]+=x[i][j];
        }
        mu[j]/=m;

        // sd
        for (int i=0; i<m; ++i) {
            sd[j]+=pow(x[i][j]-mu[j],2);
        }
        sd[j]/=m;
        sd[j]=sqrt(sd[j]);

        // data norm
        for (int i=0; i<m; ++i) {
            x[i][j]=(x[i][j]-mu[j])/sd[j];
        }
    }
}

double predict(vec<double> x, vec<vec<double>> wt, double b) {
    double res=b;
    for (int i=0; i<nf; ++i) {
        for (int p=0; p<nd; ++p) {
            res+=wt[i][p]*pow(x[i],p+1);
        }
        /* res+=wt[i]*x[i]; */
    }

    return res;
}

double cost(vec<vec<double>> tx, vec<double> ty, vec<double> py) {
    double res=0;
    for (int i=0; i<m; ++i) {
        res+=pow(py[i]-ty[i],2);
    }
    res/=m;

    return res;
}

void descend(const vec<vec<double>> &tx, const vec<double> &ty, const vec<double> &py, vec<vec<double>> &wt, double &b) {
    for (int j=0; j<nf; ++j) {
        for (int p=0; p<nd; ++p) {
            double dwjp=0;
            for (int i=0; i<m; ++i) {
                dwjp+=(py[i]-ty[i])*pow(tx[i][j],p+1);
            }
            dwjp/=m;

            wt[j][p]-=dwjp*a;
            /* if (isnan(wt[j][p])) { */
            /*     printf("detected nan\n"); */
            /*     exit(1); */
            /* } */
        }
    }

    double db=0;
    for (int i=0; i<m; ++i) {
        db+=py[i]-ty[i];
    }
    db/=m;

    b-=db*a;
}

int main() {
    srand(time(0));

    // train x, train y
    vec<vec<double>> tx, testx;
    vec<double> ty, testy;
    input(tx,ty,testx,testy);

    sort(all(testx),[](vec<double> &a, vec<double> &b){return a[14]<b[14];});

    printf("data processed\n");

    // scale
    vec<double> mu(nf), sd(nf);
    feature_scale(tx,mu,sd);

    vec<double> tmu(nf), tsd(nf);
    feature_scale(testx,tmu,tsd);

    // descent
    vec<vec<double>> wt(nf);
    for (int i=0; i<nf; ++i) wt[i].resize(nd);
    double b=0;

    for (int i=1; i<=iters; ++i) {
        vec<double> py;
        for (int i=0; i<m; ++i) {
            py.push_back(predict(tx[i],wt,b));
        }

        if (i%100==0) {
            printf("\rcost=%f (%.0f%% done)", cost(tx,ty,py), (double)i/iters*100);
            fflush(stdout);
        }

        descend(tx,ty,py,wt,b);
    }
    putchar('\n');

    int score=0;
    for (int i=0; i<20; ++i) {
        int ind=rand()%m;
        double py=predict(tx[ind],wt,b);
        printf("pred %f | real %f ", py, ty[ind]);

        int ipy=py<0.6 ? 0 : (py>1.4 ? 2 : 1);
        if (abs(py-ty[ind])<1) {
            puts("\033[1;32mOK\033[0m");
            score++;
        }
        else puts("\033[1;31mWA\033[0m");
    }

    printf("TRAIN SCORE: %d/20 (%.0f%% accuracy)\n", score, (double)score/20*100);

    vec<double> vscore(11), cnt(11);
    for (int i=0; i<m; ++i) {
        int ind=i;
        double py=predict(testx[ind],wt,b);
        printf("pred %f | real %f ", py, testy[ind]);

        int ipy=py<0.6 ? 0 : (py>1.4 ? 2 : 1);
        if (abs(py-testy[ind])<1) {
            puts("\033[1;32mOK\033[0m");
            vscore[int(testy[ind])]+=1;
        }
        else puts("\033[1;31mWA\033[0m");
        cnt[int(testy[ind])]++;
    }

    vprint(all(vscore));
    for (int i=0; i<=10; ++i) vscore[i]/=cnt[i];
    vprint(all(vscore));
}
