#include <bits/stdc++.h>
using namespace std;
#define sz(x) ((int)size(x))
#define sig(x) (x<0?-1:1)
#define all(x) begin(x),end(x)
using ll=long long;
template <typename T> using vec=vector<T>;
template <typename T> struct vec2:vector<vector<T>> {vec2()=default;vec2(int n, int m, T val=T()):vector<vector<T>>(n,vector<T>(m,val)){}};
template <typename T> void vprint(T st, T nd) {auto it=st; while(next(it)!=nd){cout<<*it<<' '; it=next(it);}cout<<*it<<'\n';}

// # training points
const int M=43;

// # distributions
const int N=30;
// # features
const int NF=3;

// learning rate
const double A=1.;

// iters
const int ITERS=1e3;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

double normpdf(double x, double mu, double sd) {
    return (1./(sd*sqrt(2.*3.14))) * exp(-0.5*pow((x-mu)/sd,2));
}

// x: x values
// ty: training y
// py: predict y
// wt: weights
// adjusts wt
void descend(const vec<double> &x, const vec<double> &ty, const vec<double> &py, vec<vec<double>> &wt) {
    for (int j=0; j<N; ++j) {
        // mean
        double dmu=0;
        for (int i=0; i<M; ++i) {
            dmu+=(ty[i]-py[i]) * wt[j][2]*normpdf(x[i], wt[j][0], wt[j][1]) * (x[i]-wt[j][0])/(wt[j][1]*wt[j][1]);
        }
        dmu*=-2.;

        // sd
        double dsd=0;
        for (int i=0; i<M; ++i) {
            dsd+=(ty[i]-py[i]) * wt[j][2]*normpdf(x[i], wt[j][0], wt[j][1]) * (pow(x[i]-wt[j][0],2)/pow(wt[j][1],3) - 1./wt[j][1]);
        }
        dsd*=-2.;

        // amplitude
        double da=0;
        for (int i=0; i<M; ++i) {
            da+=(ty[i]-py[i])*normpdf(x[i], wt[j][0], wt[j][1]);
        }
        da*=-2.;

        // update distribution j
        wt[j][0]-=dmu*A;
        wt[j][1]-=dsd*A;
        wt[j][2]-=da*A;
    }
}

vec<double> predict(const vec<double> &x, const vec<vec<double>> &wt) {
    vec<double> pred;
    for (int j=0; j<M; ++j) {
        double predj=0;
        for (auto &w:wt)
            predj+=w[2]*normpdf(x[j], w[0], w[1]);
        pred.push_back(predj);
    }

    return pred;
}

double loss(const vec<double> &tx, const vec<double> &ty, const vec<vec<double>> &wt) {
    vec<double> py=predict(tx, wt);
    double res=0;
    for (int i=0; i<M; ++i) {
        res+=pow(py[i]-ty[i],2);
    }
    res/=M;

    return res;
}

// tx, ty should be len M
// updates tx, ty
void input(vec<double> &tx, vec<double> &ty) {
    ifstream ifs("data/agefreq.csv");
    // skip labels line
    string buf;
    getline(ifs,buf);

    for (int i=0; i<M; ++i) {
        double x,freq,freqp;
        char c;
        ifs>>x>>c>>freq>>c>>freqp;

        tx[i]=x;
        ty[i]=freqp;
    }
}

int main() {
    // train x, train y
    vec<double> tx(M), ty(M);
    input(tx,ty);

    // weights (mean, sd)
    vec<vec<double>> wt(N);
    for (int i=0; i<N; ++i) {
        wt[i]={(double)(rand()%((int)*max_element(all(tx)))), (double)(rand()%5+1), (double)(rand()%100+1)/100.};
    }

    // fitting
    for (int i=1; i<=ITERS; ++i) {
        /* if (i%int(1e4)==0) { */
        /*     printf("\r%.2f%% done | loss=%e", (double)i/ITERS*100., loss(tx,ty,wt)); */
        /*     fflush(stdout); */
        /* } */

        // current prediction
        vec<double> py=predict(tx,wt);

        // gradient descent
        descend(tx, ty, py, wt);
    }

    // results
    for (int i=0; i<N; ++i) {
        printf("mu=%f sd=%f a=%f\n", wt[i][0], wt[i][1], wt[i][2]);
    }

    printf("loss=%f\n", loss(tx,ty,wt));

    vec<double> py=predict(tx,wt);
    vec<double> err;
    for (int i=0; i<M; ++i) {
        err.push_back(abs(py[i]-ty[i])/ty[i]);
    }

    vprint(all(err));
}
