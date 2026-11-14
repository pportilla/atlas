// Compute the normal fan of conv(support) + R^n_{>=0}, then a compatible
// unimodular subdivision. The process is entirely in exact lattices.
#ifndef ATLAS_SMALL_NEWTON_FAN
#include <libnormaliz/libnormaliz.h>
#endif
#include <gmpxx.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <json/json.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Z = mpz_class;
using Ray = std::vector<Z>;
using Rays = std::vector<Ray>;
using Face = std::vector<int>;
using JsonValue = Json::Value;
constexpr size_t max_rays = 16000, max_cone_records = 120000;
constexpr long max_coordinate = 100000000;

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

struct Cell { Face rays; Z index; bool active = true; };
struct Candidate { Ray point; Face support; };

class Fan {
    int n;
    bool exhaustive;
    Rays rays;
    std::map<Ray,int> index;
    std::vector<Cell> cells;
    std::vector<std::set<int>> incident;
    std::priority_queue<std::pair<Z,int>> singular;
    int newton_ray_count = 0, newton_vertices = 0;

    int id(const Ray& r) {
#ifdef ATLAS_SMALL_NEWTON_FAN
        if (*std::max_element(r.begin(), r.end()) > 64) throw std::length_error("small ray budget");
#endif
        if(!exhaustive && *std::max_element(r.begin(),r.end())>max_coordinate)
            throw std::runtime_error("Newton fan exceeded its ray-coordinate budget.");
        auto [it, fresh]=index.emplace(r,static_cast<int>(rays.size()));
        if (fresh) {
            if (rays.size()>=max_rays) throw std::runtime_error("Newton fan exceeded its ray budget.");
            rays.push_back(r); incident.emplace_back();
        }
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
        if(cells.size()>=max_cone_records) throw std::runtime_error("Newton fan exceeded its cone budget.");
        Matrix m(n); matrix(m,c); fmpz_t det; fmpz_init(det); fmpz_mat_det(det,m.a);
        Z d=integer(det); fmpz_clear(det);
        if(!d) throw std::logic_error("Triangulation produced a zero-volume cone.");
        if(d<0) {std::swap(c[n-2],c[n-1]); d=-d;}
#ifdef ATLAS_SMALL_NEWTON_FAN
        if (d > 8) throw std::length_error("small index budget");
#endif
        const int k=static_cast<int>(cells.size());
        for(int i:c) incident[i].insert(k);
        cells.push_back({std::move(c),d,true});
        if(d>1) singular.emplace(d,-k);
        return d;
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

    Candidate interior_lattice_point(const Cell& cell) const {
        Matrix a(n), inv(n); matrix(a,cell.rays);
        fmpz_t denominator; fmpz_init(denominator);
        const int ok=fmpz_mat_inv(inv.a,denominator,a.a);
        Z den=integer(denominator); fmpz_clear(denominator);
        if(!ok) throw std::logic_error("Singular lattice basis.");
        if(den<0) {den=-den; fmpz_mat_neg(inv.a,inv.a);}
        Candidate best; Z best_sum,best_max;
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
                if(!best.point.empty() && (score>best_sum || (score==best_sum && maximum>=best_max))) continue;
                Ray r(n); Face support;
                for(int j=0;j<n;++j) {
                    r[j]=point[j]/common;
                    if(num[j]!=0) support.push_back(cell.rays[j]);
                }
                best={r,support}; best_sum=score; best_max=maximum;
            }
        }
        if(best.support.size()<2) throw std::logic_error("No nontrivial lattice residue.");
        return best;
    }
    void subdivide(int k) {
        auto cand=interior_lattice_point(cells[k]);
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
            Cell old=cells[chart]; cells[chart].active=false;
            for(int i:old.rays) incident[i].erase(chart);
            for(int s:cand.support) {
                Face child=old.rays;
                *std::find(child.begin(),child.end(),s)=new_ray;
                if(add_cell(child)>=old.index) throw std::logic_error("Stellar subdivision failed to decrease index.");
            }
        }
    }
public:
    Fan(int dimension,bool exhaustive):n(dimension),exhaustive(exhaustive) {
        for(int j=0;j<n;++j) {Ray e(n);e[j]=1;id(e);}
    }
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
    Fan fan(n,exhaustive); fan.build(support); return fan.report();
}
} // namespace
int main() {
    JsonValue input,output; int rc=0;
    try {std::cin>>input;output=compute(input);}
    catch(const std::exception& e) {output["error"]=e.what();rc=1;}
    Json::StreamWriterBuilder writer; writer["indentation"]="";
    std::cout<<Json::writeString(writer,output)<<"\n"; return rc;
}
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
