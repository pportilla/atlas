// Compute the normal fan of conv(support) + R^n_{>=0}, then a compatible
// unimodular subdivision. The process is entirely in exact lattices.
//
// A singular simplicial cone is subdivided at a lattice point of its
// fundamental parallelepiped. The point decides how many cones the regular
// fan has, so it is chosen from all the points of the parallelepiped: by
// enumerating them for a cone of small index, and among the Hilbert basis
// of the cone for a larger one. The number of cones is bounded by the memory
// the system has available, not by a fixed count.
#ifndef ATLAS_SMALL_NEWTON_FAN
#include <libnormaliz/libnormaliz.h>
#endif
#include <gmpxx.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <json/json.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include <csignal>
#include <sys/time.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

namespace {
using Z = mpz_class;
using Ray = std::vector<Z>;
using Rays = std::vector<Ray>;
using Face = std::vector<int>;
using JsonValue = Json::Value;
constexpr long max_coordinate = 100000000;
// Every lattice point of the fundamental parallelepiped is enumerated for a
// cone of at most this index. A larger cone uses its Hilbert basis.
constexpr long enumeration_limit = 4096;
// A fan with at least this many cones is compared with the fan of the second
// scoring rule, and the smaller of the two is kept.
constexpr size_t comparison_threshold = 4096;
// Memory taken by one cone record with its share of the rays, the incidence
// sets and the answer, measured at about 1 kB on fans of 10^5 to 10^6 cones.
constexpr size_t record_bytes = 2048;
// Processor seconds one Hilbert basis may take. A basis is as large as the
// arithmetic of its cone makes it, up to the index, and most take
// milliseconds. Past this time the cone is subdivided at the sampled point
// instead, which needs no basis. After this many such cones the fan stops
// asking for Hilbert bases.
constexpr long hilbert_seconds = 2;
constexpr int hilbert_refusals = 3;

#ifndef ATLAS_SMALL_NEWTON_FAN
// Bytes that the fan may take: a quarter of the memory reported as available
// when the program starts.
size_t memory_allowance() {
    static const size_t allowance=[] {
        std::ifstream meminfo("/proc/meminfo");
        for(std::string key; meminfo >> key;) {
            unsigned long long kilobytes;
            if(key=="MemAvailable:" && meminfo >> kilobytes) return static_cast<size_t>(kilobytes)*1024/4;
            meminfo.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        }
        const long pages=sysconf(_SC_PHYS_PAGES), size=sysconf(_SC_PAGE_SIZE);
        return pages>0 && size>0 ? static_cast<size_t>(pages)/8*static_cast<size_t>(size) : size_t(1)<<30;
    }();
    return allowance;
}
// Records that fit in the allowance. Two fans exist while they are compared.
size_t record_allowance() {return memory_allowance()/record_bytes/2;}
#else
// The fixed-work fan declines at 64 records.
size_t record_allowance() {return 64;}
#endif

Z dot(const Ray& a, const Ray& b) {
    Z value;
    for(size_t j=0;j<a.size();++j) mpz_addmul(value.get_mpz_t(),a[j].get_mpz_t(),b[j].get_mpz_t());
    return value;
}
JsonValue array(const Ray& vec) {
    JsonValue ans(Json::arrayValue);
    for(const auto& x:vec) {
        if(x.fits_slong_p()) ans.append(Json::Int64(x.get_si()));
        else ans.append(x.get_str());
    }
    return ans;
}
JsonValue ids(const Face& face) {
    JsonValue ans(Json::arrayValue); for (int i:face) ans.append(i); return ans;
}
Z integer(const fmpz* x) {
    Z z; fmpz_get_mpz(z.get_mpz_t(),x); return z;
}
struct Matrix {
    fmpz_mat_t a;
    explicit Matrix(int n) {fmpz_mat_init(a,n,n);}
    ~Matrix() {fmpz_mat_clear(a);}
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
};

// The candidates of a cell are lattice points of its half-open fundamental
// parallelepiped, as indices into the fan's table of points. They are the
// Hilbert basis of the cell (complete) or the ones inherited from its parent.
// `sampled`: a Hilbert basis of this cell or of an ancestor took too long.
struct Cell {
    Face rays; Z index; bool active = true; std::vector<int> candidates; bool complete = false, sampled = false;
};
// A subdivision point with the face that carries it. The children that
// replace a ray of that face have the indices `sum` adds up, `largest` among them.
struct Candidate { Ray point; Face support; Z sum, largest; };
// The second scoring rule produced more cones than the fan it is compared with.
struct Larger {};

#ifndef ATLAS_SMALL_NEWTON_FAN
// Normaliz leaves a computation with an exception once this flag is set.
// The timer counts the processor time of this program, not the wall clock,
// so a busy machine does not change which bases are computed.
void interrupt_normaliz(int) {libnormaliz::nmz_interrupted=1;}
struct HilbertTime {
    HilbertTime() {
        libnormaliz::nmz_interrupted=0;
        std::signal(SIGVTALRM,interrupt_normaliz);
        itimerval timer{};
        timer.it_value.tv_sec=hilbert_seconds;
        setitimer(ITIMER_VIRTUAL,&timer,nullptr);
    }
    ~HilbertTime() {
        itimerval off{};
        setitimer(ITIMER_VIRTUAL,&off,nullptr);
        libnormaliz::nmz_interrupted=0;
    }
};
#endif

class Fan {
    int n;
    bool exhaustive;
    // Rule 0 minimizes the sum of the indices of the children. Rule 1 first
    // multiplies that sum by the number of children, which favours a point on
    // a face of low dimension.
    int rule;
    size_t cone_cap, record_limit;
    size_t active = 0;
    Rays rays;
    std::map<Ray,int> index;
    std::vector<Cell> cells;
    std::vector<std::set<int>> incident;
    std::priority_queue<std::pair<Z,int>> singular;
    int newton_ray_count = 0, newton_vertices = 0;
    Rays points;
    std::map<Ray,int> point_index;
#ifndef ATLAS_SMALL_NEWTON_FAN
    // Shared by the fans of one request: the second fan does not wait again
    // for the bases the first one gave up.
    static inline int refusals = 0;
#endif

    int id(const Ray& r) {
#ifdef ATLAS_SMALL_NEWTON_FAN
        if (*std::max_element(r.begin(), r.end()) > 64) throw std::length_error("small ray budget");
#endif
        if(!exhaustive && *std::max_element(r.begin(),r.end())>max_coordinate)
            throw std::runtime_error("Newton fan exceeded its ray-coordinate budget.");
        auto [it, fresh]=index.emplace(r,static_cast<int>(rays.size()));
        if (fresh) {rays.push_back(r); incident.emplace_back();}
        return it->second;
    }
    void matrix(Matrix& m,const Face& c) const {
        for(int row=0;row<n;++row) for(int col=0;col<n;++col)
            fmpz_set_mpz(fmpz_mat_entry(m.a,row,col), rays[c[col]][row].get_mpz_t());
    }
    Z add_cell(Face c) {
#ifdef ATLAS_SMALL_NEWTON_FAN
        if (cells.size() >= 64) throw std::length_error("small fan budget");
#endif
        if(cells.size()>=record_limit)
            throw std::runtime_error("The regular Newton fan needs more memory than is available.");
        Matrix m(n); matrix(m,c); fmpz_t det; fmpz_init(det); fmpz_mat_det(det,m.a);
        Z d=integer(det); fmpz_clear(det);
        if(!d) throw std::logic_error("Triangulation produced a zero-volume cone.");
        if(d<0) {std::swap(c[n-2],c[n-1]); d=-d;}
#ifdef ATLAS_SMALL_NEWTON_FAN
        if (d > 8) throw std::length_error("small index budget");
#endif
        const int k=static_cast<int>(cells.size());
        for(int i:c) incident[i].insert(k);
        Cell cell; cell.rays=std::move(c); cell.index=d;
        cells.push_back(std::move(cell));
        if(d>1) singular.emplace(d,-k);
        // Subdivisions only add cones, so a fan above the cap stays above it.
        if(++active>cone_cap) throw Larger{};
        return d;
    }
    void retire(int k) {
        Cell& cell=cells[k];
        cell.active=false; --active;
        for(int i:cell.rays) incident[i].erase(k);
        std::vector<int>().swap(cell.candidates);
    }

    // Pulling triangulation. A face uses its own first ray in the same global
    // order, so both sides of every shared face restrict to the same triangulation.
    // The facets of a face are its maximal proper intersections with the
    // supporting hyperplanes of the original full-dimensional normal cone.
    void pull(const Face& face, int dim, const std::vector<Face>& facets, Face prefix) {
        if(static_cast<int>(face.size())==dim) {
            prefix.insert(prefix.end(),face.begin(),face.end()); add_cell(prefix); return;
        }
        const int first=face.front();
        // Any intersection containing first cannot give a facet opposite first.
        std::set<Face> intersections;
        for(const auto& h:facets) {
            if(std::binary_search(h.begin(),h.end(),first)) continue;
            Face cut; std::set_intersection(face.begin(),face.end(),h.begin(),h.end(),std::back_inserter(cut));
            if(static_cast<int>(cut.size())>=dim-1) intersections.insert(cut);
        }
        prefix.push_back(first);
        for(const auto& cut:intersections) {
            bool maximal=true;
            for(const auto& larger:intersections)
                if(larger.size()>cut.size() && std::includes(larger.begin(),larger.end(),cut.begin(),cut.end())) {
                    maximal=false; break;
                }
            // Check dimension as well: a lower-dimensional intersection with
            // many rays can live only inside facets containing the pivot.
            if(!maximal) continue;
            if(static_cast<int>(cut.size())>dim-1 || dim>2) {
                fmpz_mat_t test; fmpz_mat_init(test,cut.size(),n);
                for(size_t i=0;i<cut.size();++i) for(int j=0;j<n;++j)
                    fmpz_set_mpz(fmpz_mat_entry(test,i,j),rays[cut[i]][j].get_mpz_t());
                const auto rank=fmpz_mat_rank(test); fmpz_mat_clear(test);
                if(rank!=dim-1) continue;
            }
            pull(cut,dim-1,facets,prefix);
        }
    }

    // index * A^{-1}, where the columns of A are the rays of the cell. Row j
    // gives the numerator over the index of the coordinate along ray j.
    void scaled_inverse(Matrix& inv,const Cell& cell) const {
        Matrix a(n); matrix(a,cell.rays);
        fmpz_t denominator,factor; fmpz_init(denominator); fmpz_init(factor);
        const int ok=fmpz_mat_inv(inv.a,denominator,a.a);
        if(ok) {
            // The denominator divides the determinant and can be smaller.
            fmpz_set_mpz(factor,cell.index.get_mpz_t());
            fmpz_divexact(factor,factor,denominator);
            fmpz_mat_scalar_mul_fmpz(inv.a,inv.a,factor);
        }
        fmpz_clear(denominator); fmpz_clear(factor);
        if(!ok) throw std::logic_error("Singular lattice basis.");
    }
    // Smaller is better. The rule's leading entry, then the sum of the child
    // indices and the largest of them.
    std::tuple<Z,Z,Z> key(const Z& sum,const Z& largest,size_t children) const {
        return {rule==1 ? Z(sum*static_cast<unsigned long>(children)) : sum,sum,largest};
    }
    std::tuple<Z,Z,Z> key(const Candidate& c) const {return key(c.sum,c.largest,c.support.size());}

    Candidate sampled_point(const Cell& cell) const {
        Matrix a(n), inv(n); matrix(a,cell.rays);
        fmpz_t denominator; fmpz_init(denominator);
        const int ok=fmpz_mat_inv(inv.a,denominator,a.a);
        Z den=integer(denominator); fmpz_clear(denominator);
        if(!ok) throw std::logic_error("Singular lattice basis.");
        if(den<0) {den=-den; fmpz_mat_neg(inv.a,inv.a);}
        Candidate best;
        // Images of the coordinate vectors generate Z^n/<rays>. If index>1
        // at least one has a nonzero residue. Its fractional coordinates give
        // a lattice point in the half-open fundamental parallelepiped. Sampling
        // extra small multiples only improves the choice, never the proof.
        for(int k=0;k<n;++k) {
            Ray generator(n),num(n);
            for(int j=0;j<n;++j) generator[j]=integer(fmpz_mat_entry(inv.a,j,k));
            for(int multiple=1;multiple<=96;++multiple) {
                Z sum=0,largest=0;
                for(int j=0;j<n;++j) {
                    num[j]+=generator[j]; mpz_mod(num[j].get_mpz_t(),num[j].get_mpz_t(),den.get_mpz_t());
                    sum+=num[j]; largest=std::max(largest,num[j]);
                }
                if(!sum) break;
                Ray point(n); Z common=0;
                for(int row=0;row<n;++row) {
                    for(int j=0;j<n;++j) point[row]+=rays[cell.rays[j]][row]*num[j];
                    if(!mpz_divisible_p(point[row].get_mpz_t(),den.get_mpz_t()))
                        throw std::logic_error("A lattice residue yielded a nonintegral point.");
                    point[row]/=den;
                    mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),point[row].get_mpz_t());
                }
                // Compare fractions after replacing the point by its primitive
                // lattice generator. Use cross products: adjugate entries need
                // not be primitive when the inverse denominator is reduced.
                Z scale=cell.index/den;
                Z score=sum*scale/common, maximum=largest*scale/common;
                if(!best.point.empty() && (score>best.sum || (score==best.sum && maximum>=best.largest))) continue;
                Ray r(n); Face support;
                for(int j=0;j<n;++j) {
                    r[j]=point[j]/common;
                    if(num[j]!=0) support.push_back(cell.rays[j]);
                }
                best={r,support,score,maximum};
            }
        }
        if(best.support.size()<2) throw std::logic_error("No nontrivial lattice residue.");
        return best;
    }

    // The best point among all the lattice points of the half-open
    // fundamental parallelepiped of a cell of small index. The row lattice of
    // the rays has a triangular Hermite basis, so the vectors x with
    // 0 <= x_i < h_ii represent Z^n/<rays> once each. The numerators of the
    // coordinates of x are index*A^{-1}x reduced modulo the index.
    void enumerated_point(Candidate& best,const Cell& cell) const {
        const long modulus=cell.index.get_si();
        Matrix inv(n); scaled_inverse(inv,cell);
        fmpz_mat_t generators,hermite; fmpz_mat_init(generators,n,n); fmpz_mat_init(hermite,n,n);
        for(int row=0;row<n;++row) for(int col=0;col<n;++col)
            fmpz_set_mpz(fmpz_mat_entry(generators,row,col),rays[cell.rays[row]][col].get_mpz_t());
        fmpz_mat_hnf(hermite,generators);
        std::vector<long> period(n);
        for(int i=0;i<n;++i) period[i]=fmpz_get_si(fmpz_mat_entry(hermite,i,i));
        fmpz_mat_clear(generators); fmpz_mat_clear(hermite);
        std::vector<std::vector<long>> step(n,std::vector<long>(n));
        fmpz_t reduced; fmpz_init(reduced);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
            fmpz_mod_ui(reduced,fmpz_mat_entry(inv.a,j,i),modulus);
            step[i][j]=fmpz_get_si(reduced);
        }
        fmpz_clear(reduced);
        auto point_of=[&](const std::vector<long>& numerators) {
            Ray point(n);
            for(int row=0;row<n;++row) {
                for(int j=0;j<n;++j) if(numerators[j]) point[row]+=rays[cell.rays[j]][row]*numerators[j];
                if(!mpz_divisible_p(point[row].get_mpz_t(),cell.index.get_mpz_t()))
                    throw std::logic_error("A lattice residue yielded a nonintegral point.");
                point[row]/=cell.index;
            }
            return point;
        };
        // The sampled point is the incumbent. Only a better key replaces it.
        long best_sum=best.sum.get_si(),best_largest=best.largest.get_si();
        long best_lead=rule==1 ? best_sum*static_cast<long>(best.support.size()) : best_sum;
        // partial[i] is the residue of the digits x_i, x_{i+1}, ...
        std::vector<std::vector<long>> partial(n+1,std::vector<long>(n));
        std::vector<long> digit(n),chosen;
        Ray chosen_point;
        while(true) {
            int i=0;
            while(i<n && digit[i]+1==period[i]) {digit[i]=0; ++i;}
            if(i==n) break;
            ++digit[i];
            for(int j=0;j<n;++j) {
                partial[i][j]+=step[i][j];
                if(partial[i][j]>=modulus) partial[i][j]-=modulus;
            }
            for(int lower=i-1;lower>=0;--lower) partial[lower]=partial[i];
            const auto& numerators=partial[0];
            long sum=0,largest=0,children=0;
            for(long v:numerators) {sum+=v; largest=std::max(largest,v); children+=v!=0;}
            if(children<2) continue;
            const long lead=rule==1 ? sum*children : sum;
            if(lead>best_lead || (lead==best_lead && (sum>best_sum || (sum==best_sum && largest>best_largest))))
                continue;
            if(lead==best_lead && sum==best_sum && largest==best_largest) {
                // Equal keys: keep the sampled point, or the smallest point found here.
                if(chosen.empty()) continue;
                Ray point=point_of(numerators);
                if(!(point<chosen_point)) continue;
                chosen_point=std::move(point);
            } else chosen_point=point_of(numerators);
            chosen=numerators; best_lead=lead; best_sum=sum; best_largest=largest;
        }
        if(chosen.empty()) return;
        // A point with a minimal key is primitive: the primitive vector on
        // its ray lies in the same parallelepiped with a smaller key.
        Z common=0;
        for(const auto& v:chosen_point) mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),v.get_mpz_t());
        if(common!=1) throw std::logic_error("A minimal lattice point was not primitive.");
        best.point=std::move(chosen_point); best.support.clear();
        for(int j=0;j<n;++j) if(chosen[j]) best.support.push_back(cell.rays[j]);
        best.sum=best_sum; best.largest=best_largest;
    }

    // The listed points that lie in the half-open parallelepiped of the cell.
    std::vector<int> inside(const std::vector<int>& list,const Cell& cell) const {
        std::vector<int> kept;
        if(list.empty()) return kept;
        Matrix inv(n); scaled_inverse(inv,cell);
        Rays rows(n,Ray(n));
        for(int j=0;j<n;++j) for(int i=0;i<n;++i) rows[j][i]=integer(fmpz_mat_entry(inv.a,j,i));
        for(int p:list) {
            bool within=true;
            for(int j=0;j<n && within;++j) {
                const Z numerator=dot(rows[j],points[p]);
                within=numerator>=0 && numerator<cell.index;
            }
            if(within) kept.push_back(p);
        }
        return kept;
    }
#ifndef ATLAS_SMALL_NEWTON_FAN
    int point_id(const Ray& point) {
        auto [it,fresh]=point_index.emplace(point,static_cast<int>(points.size()));
        if(fresh) points.push_back(point);
        return it->second;
    }
    // A lattice point of a simplicial cone with a coordinate >= 1 is the sum
    // of a ray and a lattice point of the cone. So the Hilbert basis consists
    // of the rays and of points of the half-open parallelepiped, and it
    // contains every point there that minimizes a positive linear form.
    void hilbert_candidates(Cell& cell) {
        Matrix inv(n); scaled_inverse(inv,cell);
        Rays inequalities(n,Ray(n));
        for(int j=0;j<n;++j) {
            Z common=0;
            for(int i=0;i<n;++i) {
                inequalities[j][i]=integer(fmpz_mat_entry(inv.a,j,i));
                mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),inequalities[j][i].get_mpz_t());
            }
            for(auto& entry:inequalities[j]) entry/=common;
        }
        // The dual algorithm works from the n facets. It never enumerates
        // the parallelepiped, whose size is the index.
        std::vector<int> basis;
        try {
            HilbertTime limited;
            libnormaliz::Cone<Z> cone(libnormaliz::Type::inequalities,inequalities);
            cone.compute(libnormaliz::ConeProperty::HilbertBasis,libnormaliz::ConeProperty::DualMode);
            for(const auto& element:cone.getHilbertBasis()) basis.push_back(point_id(element));
        } catch(const libnormaliz::InterruptException&) {
            // The sampled point subdivides this cone and the large cones inside it.
            cell.sampled=true; ++refusals;
            return;
        }
        cell.candidates=inside(basis,cell);
        cell.complete=true;
    }
    // The best Hilbert basis element of a cell of large index.
    void basis_point(Candidate& best,int k) {
        if(cells[k].candidates.empty() && !cells[k].complete) {
            if(cells[k].sampled || refusals>=hilbert_refusals) return;
            hilbert_candidates(cells[k]);
        }
        const Cell& cell=cells[k];
        Matrix inv(n); scaled_inverse(inv,cell);
        Rays rows(n,Ray(n));
        for(int j=0;j<n;++j) for(int i=0;i<n;++i) rows[j][i]=integer(fmpz_mat_entry(inv.a,j,i));
        auto best_key=key(best);
        int chosen=-1; Ray chosen_numerators,numerators(n);
        for(int p:cell.candidates) {
            Z sum=0,largest=0; size_t children=0;
            for(int j=0;j<n;++j) {
                numerators[j]=dot(rows[j],points[p]);
                sum+=numerators[j]; largest=std::max(largest,numerators[j]); children+=numerators[j]!=0;
            }
            if(children<2) continue;
            const auto here=key(sum,largest,children);
            // Equal keys: keep the sampled point, or the smallest point found here.
            if(here>best_key || (here==best_key && (chosen<0 || !(points[p]<points[chosen])))) continue;
            chosen=p; best_key=here; chosen_numerators=numerators;
            best.sum=sum; best.largest=largest;
        }
        if(chosen<0) return;
        Z common=0;
        for(const auto& v:points[chosen]) mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),v.get_mpz_t());
        if(common!=1) throw std::logic_error("A Hilbert basis element was not primitive.");
        best.point=points[chosen]; best.support.clear();
        for(int j=0;j<n;++j) if(chosen_numerators[j]!=0) best.support.push_back(cell.rays[j]);
    }
#endif
    // The sampled point is kept unless the complete search finds a point
    // with a strictly better key, so a fan the samples already served is unchanged.
    Candidate subdivision_point(int k) {
        Candidate best=sampled_point(cells[k]);
        if(cells[k].index<=enumeration_limit) enumerated_point(best,cells[k]);
#ifndef ATLAS_SMALL_NEWTON_FAN
        else basis_point(best,k);
#endif
        return best;
    }
    void subdivide(int k) {
        auto cand=subdivision_point(k);
        size_t old_size=rays.size(); const int new_ray=id(cand.point);
        if(rays.size()==old_size) throw std::logic_error("A subdivision ray was already in the fan.");
        auto smallest=std::min_element(cand.support.begin(),cand.support.end(),[&](int a,int b){
            return incident[a].size()<incident[b].size();
        });
        Face star;
        for(int chart:incident[*smallest]) {
            bool contains=true;
            for(int s:cand.support) if(!incident[s].count(chart)) {contains=false; break;}
            if(contains) star.push_back(chart);
        }
        // Use the complete star, including adjacent cones on the original
        // boundary. Each replacement index is the old index times a positive
        // barycentric coefficient <1, hence all affected indices decrease.
        for(int chart:star) {
            const Face old_rays=cells[chart].rays; const Z old_index=cells[chart].index;
            const std::vector<int> inherited=std::move(cells[chart].candidates);
            const bool sampled=cells[chart].sampled;
            retire(chart);
            for(int s:cand.support) {
                Face child=old_rays;
                *std::find(child.begin(),child.end(),s)=new_ray;
                const Z child_index=add_cell(child);
                if(child_index>=old_index) throw std::logic_error("Stellar subdivision failed to decrease index.");
                // A large child starts from the parent's points that it contains.
                if(child_index>enumeration_limit) {
                    cells.back().candidates=inside(inherited,cells.back());
                    cells.back().sampled=sampled;
                }
            }
        }
    }
public:
    // rule: the scoring rule. cone_cap: stop with Larger once the fan has more cones.
    Fan(int dimension,bool exhaustive,int rule=0,size_t cone_cap=std::numeric_limits<size_t>::max()):
            n(dimension),exhaustive(exhaustive),rule(rule),cone_cap(cone_cap),
            record_limit(record_allowance()) {
        for(int j=0;j<n;++j) {Ray e(n);e[j]=1;id(e);}
    }
    size_t cones() const {return active;}
#ifndef ATLAS_SMALL_NEWTON_FAN
    void build(const Rays& support) {
        // The positive-height section of this pointed cone is the Newton
        // polyhedron. Its nonhorizontal generating rays give precisely its
        // lattice vertices. Its facet equations give the normal rays.
        Rays generators=support;
        for(auto& row:generators) row.push_back(1);
        for(int j=0;j<n;++j) {Ray row(n+1);row[j]=1;generators.push_back(row);}
        libnormaliz::set_thread_limit(1);
        libnormaliz::Cone<Z> hull(libnormaliz::Type::cone,generators);
        Rays vertices;
        for(const auto& r:hull.getExtremeRays()) if(r.back()>0) {
            Ray vertex(n);
            for(int j=0;j<n;++j) {
                if(!mpz_divisible_p(r[j].get_mpz_t(),r.back().get_mpz_t()))
                    throw std::logic_error("Nonintegral Newton vertex.");
                vertex[j]=r[j]/r.back();
            }
            vertices.push_back(vertex);
        }
        std::sort(vertices.begin(),vertices.end()); newton_vertices=vertices.size();
        std::map<Ray,Z> normals;
        for(const auto& h:hull.getSupportHyperplanes()) {
            Z common=0;
            for(int j=0;j<n;++j) mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),h[j].get_mpz_t());
            if(!common) continue;  // facet at infinity
            Ray r(n); for(int j=0;j<n;++j) r[j]=h[j]/common;
            Z order=dot(r,vertices.front()); for(const auto& m:vertices) order=std::min(order,dot(r,m));
            normals.emplace(r,order);
        }
        refine(vertices, normals);
    }
#else
    void build_small(const Rays& support) {
        // The normal cone at m is cut out by v >= 0 and (e - m).v >= 0.
        // An extreme ray is the signed cofactor vector of n-1 independent
        // active rows. At most 6 rows occur here in dimensions <= 4.
        Rays vertices;
        std::map<Ray,Z> normals;
        for (const auto& m : support) {
            Rays inequalities(n, Ray(n));
            for (int j = 0; j < n; ++j) inequalities[j][j] = 1;
            for (const auto& e : support) if (e != m) {
                Ray h(n);
                for (int j = 0; j < n; ++j) h[j] = e[j] - m[j];
                inequalities.push_back(std::move(h));
            }
            std::set<Ray> rays_here;
            std::vector<bool> active(inequalities.size());
            std::fill(active.end() - (n - 1), active.end(), true);
            Matrix minor(n-1);
            do {
                Ray ray(n);
                for (int skip=0; skip<n; ++skip) {
                    int row=0;
                    for(size_t i=0;i<active.size();++i) if(active[i]) {
                        int col=0;
                        for(int j=0;j<n;++j) if(j!=skip)
                            fmpz_set_mpz(fmpz_mat_entry(minor.a,row,col++), inequalities[i][j].get_mpz_t());
                        ++row;
                    }
                    fmpz_t det; fmpz_init(det); fmpz_mat_det(det,minor.a);
                    ray[skip]=integer(det); fmpz_clear(det);
                    if(skip%2) ray[skip]=-ray[skip];
                }
                const auto first=std::find_if(ray.begin(),ray.end(),[](const Z& z){return z!=0;});
                if(first==ray.end()) continue;
                if(*first<0) for(auto& z:ray) z=-z;
                if(std::any_of(inequalities.begin(),inequalities.end(),[&](const Ray& h){return dot(h,ray)<0;}))
                    continue;
                Z divisor=0;
                for(const auto& z:ray) mpz_gcd(divisor.get_mpz_t(),divisor.get_mpz_t(),z.get_mpz_t());
                for(auto& z:ray) z/=divisor;
                rays_here.insert(std::move(ray));
            } while(std::next_permutation(active.begin(),active.end()));
            // Only full-dimensional normal cones belong to vertices. The
            // lower-dimensional cones of redundant support points add no rays.
            fmpz_mat_t matrix; fmpz_mat_init(matrix,rays_here.size(),n);
            int i=0;
            for(const auto& ray:rays_here) {
                for(int j=0;j<n;++j) fmpz_set_mpz(fmpz_mat_entry(matrix,i,j),ray[j].get_mpz_t());
                ++i;
            }
            const auto rank=fmpz_mat_rank(matrix); fmpz_mat_clear(matrix);
            if(rank!=n) continue;
            vertices.push_back(m);
            for(const auto& ray:rays_here) normals.emplace(ray,dot(ray,m));
        }
        newton_vertices=vertices.size();
        refine(vertices,normals);
    }
#endif
    void refine(const Rays& vertices, const std::map<Ray,Z>& normals) {
        for(const auto& [r,order]:normals) id(r);
        newton_ray_count=rays.size();
        for(const auto& m:vertices) {
            Face cone;
            for(const auto& [r,order]:normals) if(dot(r,m)==order) cone.push_back(id(r));
            std::sort(cone.begin(),cone.end());
            if(static_cast<int>(cone.size())<n) throw std::logic_error("Newton vertex has incomplete normal cone.");
            if(static_cast<int>(cone.size())==n) {add_cell(cone);continue;}
            // H-description: orthant inequalities, and (v-m)·w>=0 for
            // every other Newton vertex v. Redundant inequalities are safe.
            std::vector<Face> facets;
            auto add_hyperplane=[&](const Ray& h) {
                Face eq;
                for(int i:cone) if(dot(rays[i],h)==0) eq.push_back(i);
                if(!eq.empty() && eq.size()<cone.size()) facets.push_back(eq);
            };
            for(int j=0;j<n;++j) {Ray e(n);e[j]=1;add_hyperplane(e);}
            for(const auto& v:vertices) if(v!=m) {
                Ray h(n); for(int j=0;j<n;++j) h[j]=v[j]-m[j]; add_hyperplane(h);
            }
            pull(cone,n,facets,{});
        }
        while(!singular.empty()) {
            const int k=-singular.top().second; singular.pop();
            if(cells[k].active) subdivide(k);
        }
    }
    JsonValue report() const {
        JsonValue result;
        result["rays"]=JsonValue(Json::arrayValue);
        result["cones"]=JsonValue(Json::arrayValue);
        result["boundary_rays"]=JsonValue(Json::arrayValue);
        for(size_t i=0;i<rays.size();++i) {
            result["rays"].append(array(rays[i]));
            if(i>=static_cast<size_t>(n) && *std::min_element(rays[i].begin(),rays[i].end())==0)
                result["boundary_rays"].append(static_cast<int>(i));
        }
        for(const auto& c:cells) if(c.active) result["cones"].append(ids(c.rays));
        result["newton_ray_count"]=newton_ray_count;
        result["newton_vertex_count"]=newton_vertices;
        result["boundary_subdivided"]=!result["boundary_rays"].empty();
        result["isomorphism_off_origin"]=result["boundary_rays"].empty();
        return result;
    }
};

JsonValue trial_supports(const Rays& support, int n) {
    JsonValue result(Json::arrayValue);
    std::set<Rays> seen;
    auto propose = [&](Rays terms) {
        std::sort(terms.begin(), terms.end());
        terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
        if (!seen.insert(terms).second) return;
        JsonValue item(Json::arrayValue);
        for (const Ray& e : terms) item.append(array(e));
        result.append(item);
    };
    Z lo = -1, hi = 0;
    for (const Ray& e : support) {
        const Z d = std::accumulate(e.begin(), e.end(), Z(0));
        if (lo < 0 || d < lo) lo = d;
        hi = std::max(hi, d);
    }
    if (hi > 32 || support.size() > 64) return result;
    std::vector<unsigned> centres;
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
        bool minimal = std::none_of(centres.begin(), centres.end(), [&](unsigned c){return (mask & c) == c;});
        if (!minimal) continue;
        bool singular = std::all_of(support.begin(), support.end(), [&](const Ray& e){
            Z d = 0; for (int j=0; j<n; ++j) if (mask & (1u<<j)) d += e[j];
            return d >= 2;
        });
        if (singular) centres.push_back(mask);
    }
    auto unit = [&](int i, const Z& a){Ray e(n); e[i]=a; return e;};
    for (int i=0; i<n; ++i) for (int j=i+1; j<n; ++j) {
        unsigned pair = (1u<<i)|(1u<<j);
        if (std::none_of(centres.begin(),centres.end(),[&](unsigned c){return (pair&c)==c;})) continue;
        for (int reverse=0; reverse<2; ++reverse) for(int d : {2,3})
            propose({unit(reverse?j:i,d),unit(reverse?i:j,1)});
    }
    for (const Z& d : std::vector<Z>{hi+1,hi,hi+2,lo+1}) {
        Rays terms=support;
        for (int i=0; i<n; ++i) terms.push_back(unit(i,d));
        propose(std::move(terms));
    }
    return result;
}

#ifndef ATLAS_SMALL_NEWTON_FAN
JsonValue compute(const JsonValue& obj) {
    const auto& terms=obj["exponents"];
    const bool exhaustive=obj.get("exhaustive",false).asBool();
    if(!terms.isArray() || terms.empty()) throw std::invalid_argument("Expected a nonempty array of exponent vectors.");
    if(!exhaustive && terms.size()>4096)
        throw std::invalid_argument("Expected 1–4096 exponent vectors.");
    const int n=terms[0].size();
    if(n<2 || n>8) throw std::invalid_argument("Newton fan dimension must be 2–8.");
    // An axial support point a*e_j already generates a*e_j + the orthant.
    // Only the lowest such power can be needed for the Newton polyhedron.
    Ray axial(n);
    Rays support;
    support.reserve(terms.size());
    for(const auto& row:terms) {
        if(!row.isArray() || static_cast<int>(row.size())!=n) throw std::invalid_argument("Inconsistent exponent dimension.");
        Ray e;
        e.reserve(n);
        int axis = -1;  // -1 for zero so far, n for a mixed monomial.
        Z total;
        for(const auto& entry:row) {
            if(!(entry.isInt64() || entry.isUInt64() || (exhaustive && entry.isString())))
                throw std::invalid_argument("Expected nonnegative integer exponents.");
            const auto digits=entry.asString();
            if(digits.empty() || digits.find_first_not_of("0123456789")!=std::string::npos)
                throw std::invalid_argument("Expected nonnegative integer exponents.");
            e.emplace_back(digits,10);
            if (axis != n && e.back() != 0)
                axis = axis < 0 ? static_cast<int>(e.size()) - 1 : n;
            if(!exhaustive) total+=e.back();
        }
        if(!exhaustive && total>128) throw std::invalid_argument("Total degree exceeds 128.");
        if (axis >= 0 && axis < n && (!axial[axis] || e[axis] < axial[axis]))
            axial[axis] = e[axis];
        support.push_back(std::move(e));
    }
    if (obj["operation"] == "trials")
        return trial_supports(support, n);
    if (std::any_of(axial.begin(), axial.end(), [](const Z& a) { return a != 0; })) {
        support.erase(std::remove_if(support.begin(), support.end(), [&](const Ray& e) {
            for (int j = 0; j < n; ++j) {
                if (!axial[j] || e[j] < axial[j]) continue;
                if (e[j] > axial[j]) return true;
                // Equality on the axis still dominates a mixed monomial.
                for (int k = 0; k < n; ++k)
                    if (k != j && e[k] != 0) return true;
                return false;
            }
            return false;
        }), support.end());
    }
    std::sort(support.begin(),support.end());support.erase(std::unique(support.begin(),support.end()),support.end());
    // The caller builds one chart on every cone and says how many fit in its
    // memory. A fan past that number is of no use to it, so it stops there.
    size_t cone_limit=std::numeric_limits<size_t>::max();
    if(obj.isMember("cone_limit")) {
        if(!obj["cone_limit"].isUInt64()) throw std::invalid_argument("Expected a nonnegative cone limit.");
        cone_limit=obj["cone_limit"].asUInt64();
    }
    std::unique_ptr<Fan> fan;
    try {
        fan=std::make_unique<Fan>(n,exhaustive,0,cone_limit); fan->build(support);
    } catch(const Larger&) {fan.reset();}
    if(!fan || fan->cones()>=comparison_threshold) {
        // The number of cones depends on the scoring rule in a way no single
        // rule wins. Building a fan costs far less than its charts, so a
        // large fan is compared with the fan of the second rule, which stops
        // as soon as it has more cones.
        try {
            Fan other(n,exhaustive,1,fan ? fan->cones()-1 : cone_limit); other.build(support);
            return other.report();
        } catch(const Larger&) {
        } catch(const std::runtime_error&) {  // a resource bound the first fan stayed within
            if(!fan) throw;
        } catch(const std::bad_alloc&) {
            if(!fan) throw;
        }
    }
    if(!fan) throw std::runtime_error("The regular Newton fan has more than "+std::to_string(cone_limit)+
                                      " cones, the number whose charts fit in the available memory.");
    return fan->report();
}
} // namespace
namespace {
// An allocation beyond the allowance fails in this process. It cannot
// exhaust the memory of the machine: a Hilbert basis can be as large as the
// index of its cone.
void bound_memory() {
    std::ifstream statm("/proc/self/statm");
    unsigned long long pages=0;
    const long size=sysconf(_SC_PAGE_SIZE);
    if(!(statm >> pages) || size<=0) return;
    struct rlimit limit;
    if(getrlimit(RLIMIT_AS,&limit)) return;
    const rlim_t bound=static_cast<rlim_t>(pages)*static_cast<rlim_t>(size)+memory_allowance();
    if(limit.rlim_cur!=RLIM_INFINITY && limit.rlim_cur<=bound) return;
    limit.rlim_cur=bound;
    setrlimit(RLIMIT_AS,&limit);
}
int program() {
    JsonValue input,output; int rc=0;
#ifdef __linux__
    // A fan has no time limit of its own. It stops with the solver that asked for it.
    prctl(PR_SET_PDEATHSIG,SIGKILL);
#endif
    try {bound_memory();std::cin>>input;output=compute(input);}
    catch(const std::bad_alloc&) {output["error"]="The regular Newton fan needs more memory than is available.";rc=1;}
    catch(const std::exception& e) {output["error"]=e.what();rc=1;}
    Json::StreamWriterBuilder writer; writer["indentation"]="";
    std::cout<<Json::writeString(writer,output)<<"\n"; return rc;
}
} // namespace

// The newton_fan program's entry, also run in a forked copy of a solver that
// already loaded this library.
extern "C" __attribute__((visibility("default"))) int atlas_newton_fan_main(int, char**) noexcept {
    const int status = program();
    std::cout.flush();
    std::fflush(nullptr);
    return status;
}

#ifndef ATLAS_HELPER_LIBRARY
int main(int argc, char** argv) { return atlas_newton_fan_main(argc, argv); }
#endif
#else
} // namespace

// Null means the isolated full fan engine must handle the input. Thread-local
// storage holds this short result until the caller copies it.
extern "C" const char* atlas_small_newton_fan(const char* bytes, bool trials) noexcept {
    try {
        std::istringstream input(bytes);
        int n, count;
        if (!(input >> n >> count) || n<2 || n>4 || count<1 || count>3) return nullptr;
        Rays support(count,Ray(n));
        for(auto& e:support) for(auto& x:e) {
            int value;
            if (!(input >> value) || value<0 || value>3) return nullptr;
            x=value;
        }
        JsonValue result;
        if (trials) {
            result=trial_supports(support,n);
        } else {
            std::sort(support.begin(),support.end());
            support.erase(std::unique(support.begin(),support.end()),support.end());
            Fan fan(n,false);
            fan.build_small(support);
            result=fan.report();
        }
        Json::StreamWriterBuilder writer; writer["indentation"]="";
        static thread_local std::string json;
        json=Json::writeString(writer,result);
        return json.c_str();
    } catch (...) { return nullptr; }
}
#endif
