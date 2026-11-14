// Compute the normal fan of conv(support) + R^n_{>=0}, then a compatible
// unimodular subdivision. The process is entirely in exact lattices.
#include <libnormaliz/libnormaliz.h>
#include <flint/fmpz_mat.h>
#include <json/json.h>
#include <algorithm>
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
using Ray = std::vector<long>;
using Rays = std::vector<Ray>;
using Face = std::vector<int>;
using BigRow = std::vector<Z>;
using JsonValue = Json::Value;
constexpr size_t max_rays = 16000, max_cone_records = 120000;
constexpr long max_coordinate = 100000000;

long coordinate(const Z& z) {
    if (z<0 || z>max_coordinate) throw std::runtime_error("Newton fan exceeded its ray-coordinate budget.");
    return z.get_si();
}
long dot(const Ray& a, const Ray& b) {
    // One argument in these calls is a monomial or a difference of monomials,
    // whose l1 norm is at most 256. The other is a bounded nonnegative ray.
    return std::inner_product(a.begin(),a.end(),b.begin(),0L);
}
JsonValue array(const std::vector<long>& vec) {
    JsonValue ans(Json::arrayValue);
    for (long x:vec) ans.append(Json::Int64(x));
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
    Rays rays;
    std::map<Ray,int> index;
    std::vector<Cell> cells;
    std::vector<std::set<int>> incident;
    std::priority_queue<std::pair<Z,int>> singular;
    int newton_ray_count = 0, newton_vertices = 0;

    int id(const Ray& r) {
        auto [it, fresh]=index.emplace(r,static_cast<int>(rays.size()));
        if (fresh) {
            if (rays.size()>=max_rays) throw std::runtime_error("Newton fan exceeded its ray budget.");
            rays.push_back(r); incident.emplace_back();
        }
        return it->second;
    }
    void matrix(Matrix& m,const Face& c) const {
        for(int row=0;row<n;++row) for(int col=0;col<n;++col)
            fmpz_set_si(fmpz_mat_entry(m.a,row,col), rays[c[col]][row]);
    }
    Z add_cell(Face c) {
        if(cells.size()>=max_cone_records) throw std::runtime_error("Newton fan exceeded its cone budget.");
        Matrix m(n); matrix(m,c); fmpz_t det; fmpz_init(det); fmpz_mat_det(det,m.a);
        Z d=integer(det); fmpz_clear(det);
        if(!d) throw std::logic_error("Triangulation produced a zero-volume cone.");
        if(d<0) {std::swap(c[n-2],c[n-1]); d=-d;}
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
                    fmpz_set_si(fmpz_mat_entry(test,i,j),rays[cut[i]][j]);
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
            BigRow generator(n),num(n);
            for(int j=0;j<n;++j) generator[j]=integer(fmpz_mat_entry(inv.a,j,k));
            for(int multiple=1;multiple<=96;++multiple) {
                Z sum=0,largest=0;
                for(int j=0;j<n;++j) {
                    num[j]+=generator[j]; mpz_mod(num[j].get_mpz_t(),num[j].get_mpz_t(),den.get_mpz_t());
                    sum+=num[j]; largest=std::max(largest,num[j]);
                }
                if(!sum) break;
                BigRow point(n); Z common=0;
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
                    r[j]=coordinate(point[j]/common);
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
    explicit Fan(int dimension):n(dimension) {
        for(int j=0;j<n;++j) {Ray e(n);e[j]=1;id(e);}
    }
    void build(const Rays& support) {
        // The positive-height section of this pointed cone is the Newton
        // polyhedron. Its nonhorizontal generating rays give precisely its
        // lattice vertices. Its facet equations give the normal rays.
        std::vector<BigRow> generators;
        for(const auto& e:support) {BigRow row(e.begin(),e.end());row.push_back(1);generators.push_back(row);}
        for(int j=0;j<n;++j) {BigRow row(n+1);row[j]=1;generators.push_back(row);}
        libnormaliz::set_thread_limit(1);
        libnormaliz::Cone<Z> hull(libnormaliz::Type::cone,generators);
        Rays vertices;
        for(const auto& r:hull.getExtremeRays()) if(r.back()>0) {
            Ray vertex(n);
            for(int j=0;j<n;++j) {
                if(!mpz_divisible_p(r[j].get_mpz_t(),r.back().get_mpz_t()))
                    throw std::logic_error("Nonintegral Newton vertex.");
                vertex[j]=coordinate(r[j]/r.back());
            }
            vertices.push_back(vertex);
        }
        std::sort(vertices.begin(),vertices.end()); newton_vertices=vertices.size();
        std::map<Ray,long> normals;
        for(const auto& h:hull.getSupportHyperplanes()) {
            Z common=0;
            for(int j=0;j<n;++j) mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),h[j].get_mpz_t());
            if(!common) continue;  // facet at infinity
            Ray r(n); for(int j=0;j<n;++j) r[j]=coordinate(h[j]/common);
            long order=dot(r,vertices.front()); for(const auto& m:vertices) order=std::min(order,dot(r,m));
            normals.emplace(r,order);
        }
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

JsonValue compute(const JsonValue& obj) {
    const auto& terms=obj["exponents"];
    if(!terms.isArray() || terms.empty() || terms.size()>4096)
        throw std::invalid_argument("Expected 1–4096 exponent vectors.");
    const int n=terms[0].size();
    if(n<2 || n>8) throw std::invalid_argument("Newton fan dimension must be 2–8.");
    Rays support;
    for(const auto& row:terms) {
        if(!row.isArray() || static_cast<int>(row.size())!=n) throw std::invalid_argument("Inconsistent exponent dimension.");
        Ray e;
        long total=0;
        for(const auto& entry:row) {
            if(!entry.isInt() || entry.asInt()<0) throw std::invalid_argument("Expected nonnegative integer exponents.");
            e.push_back(entry.asInt()); total+=e.back();
        }
        if(total>128) throw std::invalid_argument("Total degree exceeds 128.");
        support.push_back(e);
    }
    std::sort(support.begin(),support.end());support.erase(std::unique(support.begin(),support.end()),support.end());
    Fan fan(n); fan.build(support); return fan.report();
}
} // namespace
int main() {
    JsonValue input,output; int rc=0;
    try {std::cin>>input;output=compute(input);}
    catch(const std::exception& e) {output["error"]=e.what();rc=1;}
    Json::StreamWriterBuilder writer; writer["indentation"]="";
    std::cout<<Json::writeString(writer,output)<<"\n"; return rc;
}
