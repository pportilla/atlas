// Regular binomial fans, their admissible coordinate centres and orbit incidences.
// A star subdivision at the sum of a face of a regular cone is a smooth
// blowup. Summing the two original exponent vectors on its coordinate image
// tests containment in the singular locus. The chart criterion and the dense
// orbit determining each intersection are described in engine/binomial_complex.py.
#include "groebner.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

namespace {
using Integer = boost::multiprecision::cpp_int;
using Vector = std::vector<Integer>;
using Face = std::vector<size_t>;
using Fan = std::vector<Face>;
using Rays = std::vector<Vector>;

template<class T> T read(std::istream& in) {
    T x;
    if (!(in >> x)) throw std::invalid_argument("Invalid binomial fan data.");
    return x;
}
template<class T> std::vector<T> row(std::istream& in, size_t n) {
    std::vector<T> a(n);
    for (auto& x : a) x = read<T>(in);
    return a;
}
template<class T> void write(std::ostream& out, const std::vector<T>& a) {
    for (auto& x : a) out << ' ' << x;
}

// Dividing a Laurent segment by its primitive character gives the powers of
// one variable. Translation and projection onto pivot coordinates change a
// higher-dimensional face polynomial by a torus unit and a torus isogeny.
std::string newton_support(std::istream& in, size_t n, bool edge) {
    size_t count = read<size_t>(in), rank = read<size_t>(in);
    if (!count || !rank || rank > n || (edge && rank != 1))
        throw std::invalid_argument("Invalid Newton projection dimensions.");
    auto axes = row<size_t>(in, rank);
    if (*std::max_element(axes.begin(), axes.end()) >= n)
        throw std::invalid_argument("Invalid Newton projection coordinate.");
    Rays support;
    for (size_t i = 0; i < count; ++i) support.push_back(row<Integer>(in, n));
    std::ostringstream out;
    if (edge) {
        Vector direction(n);
        Integer length = 0;
        for (size_t j = 0; j < n; ++j) {
            direction[j] = support.back()[j] - support.front()[j];
            length = gcd(length, direction[j]);
        }
        if (!length || !direction[axes.front()]) throw std::invalid_argument("Invalid Newton edge.");
        for (auto& value : direction) value /= length;
        out << length; write(out, direction); out << '\n';
        for (const auto& exponent : support) {
            Integer k = (exponent[axes.front()]-support.front()[axes.front()])/direction[axes.front()];
            for (size_t j = 0; j < n; ++j)
                if (support.front()[j]+k*direction[j] != exponent[j])
                    throw std::invalid_argument("The support does not lie in the stated edge.");
            out << k << '\n';
        }
    } else {
        Vector minimum = support.front();
        for (const auto& e : support) for (auto j : axes) minimum[j] = std::min(minimum[j], e[j]);
        for (const auto& e : support) {
            for (auto j : axes) out << ' ' << e[j]-minimum[j];
            out << '\n';
        }
    }
    return out.str();
}

std::string face_coefficients(std::istream& in) {
    size_t count = read<size_t>(in);
    Vector numerators(count), denominators(count);
    Integer denominator = 1, content = 0;
    for (size_t i = 0; i < count; ++i) {
        numerators[i] = read<Integer>(in); denominators[i] = read<Integer>(in);
        if (denominators[i] <= 0) throw std::invalid_argument("Invalid Newton coefficient denominator.");
        denominator = lcm(denominator, denominators[i]);
        if (boost::multiprecision::msb(denominator) >= 8192) return "denominator\n";
    }
    for (size_t i = 0; i < count; ++i) {
        numerators[i] *= denominator/denominators[i];
        content = gcd(content, numerators[i]);
    }
    if (!content) throw std::invalid_argument("A Newton face must have a nonzero coefficient.");
    std::ostringstream out;
    for (auto& v : numerators) {
        v /= content;
        if (v && boost::multiprecision::msb(Integer(abs(v))) >= 8192) return "coefficient\n";
        out << v << '\n';
    }
    return out.str();
}
template<class F> void subsets(const Face& list, size_t k, F visit) {
    Face chosen;
    auto next = [&](auto&& self, size_t at) -> void {
        if (chosen.size() == k) { visit(chosen); return; }
        const size_t last = list.size() - (k - chosen.size());
        for (size_t i = at; i <= last; ++i) {
            chosen.push_back(list[i]); self(self, i+1); chosen.pop_back();
        }
    };
    if (k <= list.size()) next(next, 0);
}
Face sorted(Face c) { std::sort(c.begin(), c.end()); return c; }

struct Step { Face face, image; Integer a, b; };
struct Choice { Step step; Integer large, small; size_t incidence = 0; };
bool better(const Choice& a, const Choice& b) {
    if (a.large != b.large) return a.large > b.large;
    if (a.small != b.small) return a.small > b.small;
    if (a.incidence != b.incidence) return a.incidence < b.incidence;
    return a.step.face < b.step.face;
}
bool centre(const Fan& cones, const Fan& bad, const Rays& rays, const Vector& orders,
            const Vector& a, const Vector& b, Choice& result) {
    const size_t n=a.size();
    for (size_t k=2; k<=n; ++k) {
        std::map<Face,Choice> candidates;
        for (const auto& c : bad) subsets(c, k, [&](const Face& face) {
            if (candidates.count(face)) return;
            Integer plus=0, minus=0;
            for (auto i : face) {
                if (orders[i]>0) plus+=orders[i];
                else minus-=orders[i];
            }
            if (!plus || !minus) return;
            Step step{face,{},0,0};
            for(size_t j=0;j<n;++j) {
                if(std::any_of(face.begin(),face.end(),[&](size_t i){ return rays[i][j]!=0; })) {
                    step.image.push_back(j); step.a+=a[j]; step.b+=b[j];
                }
            }
            if (step.a<2 || step.b<2) return;
            candidates.emplace(face,Choice{step, std::max(plus,minus), std::min(plus,minus), 0});
        });
        if(candidates.empty()) continue;
        for(const auto& c:cones) subsets(sorted(c),k,[&](const Face& face){
            auto it=candidates.find(face);
            if(it!=candidates.end()) ++it->second.incidence;
        });
        result = std::min_element(candidates.begin(),candidates.end(),[](const auto& x,const auto& y){
            return better(x.second,y.second);
        })->second;
        return true;
    }
    return false;
}
int terminal(const Face& c,const Vector& orders,size_t n,bool intrinsic) {
    size_t plus=0,minus=0,p=0,m=0;
    for(size_t j=0;j<c.size();++j) {
        if(orders[c[j]]>0) {++plus; p=j;}
        if(orders[c[j]]<0) {++minus; m=j;}
    }
    if(!plus || !minus) return -1; // a unit term
    if(plus==1 && c[p]<n && orders[c[p]]==1) return static_cast<int>(p);
    if(minus==1 && c[m]<n && orders[c[m]]==-1) return static_cast<int>(m);
    // On u_E = a*v^d the strict transform is a graph, and its reduced
    // intersection with E is v=0. All other exceptional axes are free.
    if(intrinsic && plus==1 && minus==1) {
        if(c[p]>=n && c[m]<n && orders[c[p]]==1) return static_cast<int>(p);
        if(c[m]>=n && c[p]<n && orders[c[m]]==-1) return static_cast<int>(m);
    }
    return -2;
}
std::string fan(std::istream& in,size_t n,bool intrinsic) {
    const Integer max_steps=read<Integer>(in), max_cones=read<Integer>(in);
    const auto a=row<Integer>(in,n), b=row<Integer>(in,n);
    Rays rays(n,Vector(n));
    Face first(n);
    Vector orders(n);
    for(size_t i=0;i<n;++i){rays[i][i]=1; first[i]=i; orders[i]=a[i]-b[i];}
    Fan cones{first};
    std::vector<Step> steps;
    while(true) {
        Fan bad;
        std::vector<int> certificates;
        for(auto& c:cones) {
            int cert=terminal(c,orders,n,intrinsic);
            certificates.push_back(cert);
            if(cert==-2) bad.push_back(sorted(c));
        }
        if(bad.empty()) {
            std::ostringstream out;
            out<<rays.size()<<" "<<cones.size()<<" "<<steps.size()<<"\n";
            for(auto& r:rays){ write(out,r); out<<"\n"; }
            for(size_t ci=0;ci<cones.size();++ci) {
                out<<certificates[ci]; write(out,cones[ci]);
                for(auto r:cones[ci]) out<<" "<<orders[r];
                out<<"\n";
            }
            for(auto& s:steps) {
                out<<s.face.size(); write(out,s.face); out<<" "<<s.image.size(); write(out,s.image);
                out<<" "<<s.a<<" "<<s.b<<"\n";
            }
            return out.str();
        }
        Choice selected;
        if(steps.size()>=max_steps || !centre(cones,bad,rays,orders,a,b,selected)) return "none";
        auto& face=selected.step.face;
        if(Integer(cones.size())+Integer(selected.incidence)*(face.size()-1)>max_cones) return "none";
        size_t next=rays.size();
        Vector ray(n); Integer weight=0;
        for(auto i:face){ for(size_t j=0;j<n;++j) ray[j]+=rays[i][j]; weight+=orders[i]; }
        rays.push_back(std::move(ray)); orders.push_back(weight); steps.push_back(selected.step);
        Fan refined;
        for(const auto& c:cones) {
            const auto s=sorted(c);
            if(!std::includes(s.begin(),s.end(),face.begin(),face.end())) {refined.push_back(c);continue;}
            for(auto i:face){auto child=c; *std::find(child.begin(),child.end(),i)=next; refined.push_back(std::move(child));}
        }
        cones.swap(refined);
    }
}

using Piece = std::pair<Face, Face>;
struct ShortFirst {
    bool operator()(const Piece& a,const Piece& b) const {
        return a.first.size()!=b.first.size()?a.first.size()<b.first.size():a<b;
    }
};
struct Seen { size_t witness; bool meets; };
std::string pairings(std::istream& in, size_t n) {
    const size_t nr = read<size_t>(in), nf = read<size_t>(in);
    std::vector<Vector> forms;
    for (size_t j=0; j<nf; ++j) forms.push_back(row<Integer>(in,n));
    std::ostringstream out;
    for (size_t i=0; i<nr; ++i) {
        const auto ray = row<Integer>(in,n);
        for (const auto& f : forms)
            out << ' ' << std::inner_product(ray.begin(),ray.end(),f.begin(),Integer(0));
        out << '\n';
    }
    return out.str();
}
// Signed Laurent character and the ratio of its two coefficients. Rational
// scalars are transported as numerator/denominator pairs.
std::string character(std::istream& in, size_t n) {
    Vector left = row<Integer>(in, n), right = row<Integer>(in, n);
    Integer a = read<Integer>(in), b = read<Integer>(in);
    Integer c = read<Integer>(in), d = read<Integer>(in);
    if (!a || !b || !c || !d) throw std::invalid_argument("Expected nonzero rational coefficients.");
    Vector delta(n), direction(n);
    Integer common = 0;
    for (size_t j = 0; j < n; ++j) {
        delta[j] = left[j]-right[j];
        common = gcd(common, delta[j]);
    }
    if (!common) throw std::invalid_argument("A binomial needs distinct monomials.");
    for (size_t j = 0; j < n; ++j) direction[j] = delta[j]/common;
    Integer numerator = -c*b, denominator = d*a;
    Integer content = gcd(numerator, denominator);
    if (denominator < 0) content = -content;
    std::ostringstream out;
    out << common << ' ' << numerator/content << ' ' << denominator/content << '\n';
    write(out, delta); out << '\n';
    write(out, direction); out << '\n';
    return out.str();
}

std::string minima(std::istream& in, size_t width) {
    std::ostringstream out;
    const size_t count = read<size_t>(in);
    for (size_t i = 0; i < count; ++i) {
        const Vector values = row<Integer>(in, width);
        const auto& minimum = *std::min_element(values.begin(), values.end());
        out << minimum;
        for (size_t j = 0; j < width; ++j) if (values[j] == minimum) out << ' ' << j;
        out << '\n';
    }
    return out.str();
}

// Exceptional divisor data under a smooth blowup of the given codimension.
std::string exceptional_data(std::istream& in, size_t codimension) {
    auto count = read<size_t>(in);
    Integer multiplicity = 0, discrepancy = codimension;
    for (size_t i = 0; i < count; ++i) {
        const auto terms = row<Integer>(in, 3);
        multiplicity += terms[0] * terms[2];
        discrepancy += (terms[1] - 1) * terms[2];
    }
    return multiplicity.str() + " " + discrepancy.str();
}

// Choose the largest multiplicity of any plane trace and, for a tie, the
// last divisor label (or the last set sharing a component of order one).
std::string surface_curve_choice(std::istream& in, size_t curves, bool include_matches) {
    const size_t labels = read<size_t>(in);
    Integer maximum = 0;
    size_t last = 0;
    std::vector<size_t> shared;
    std::vector<std::vector<Integer>> rows;
    for (size_t i = 0; i < curves; ++i) {
        auto orders = row<Integer>(in, labels);
        std::vector<size_t> present;
        for (size_t j = 0; j < labels; ++j) {
            if (orders[j] < 0) throw std::invalid_argument("Negative surface trace order.");
            if (orders[j] > 0) present.push_back(j);
            if (orders[j] > maximum || (orders[j] == maximum && j > last)) {
                maximum = orders[j];
                last = j;
            }
        }
        if (present.size() > 1 && shared < present) shared = std::move(present);
        if (include_matches) rows.push_back(std::move(orders));
    }
    if (maximum > 1) shared = {last};
    if (shared.empty()) return "none";
    std::ostringstream out;
    out << maximum;
    write(out, shared);
    if (include_matches) {
        out << '\n';
        for (size_t i=0; i<rows.size(); ++i)
            if (std::all_of(shared.begin(),shared.end(),[&](size_t j){ return rows[i][j] == maximum; }))
                out << i << ' ';
    }
    return out.str();
}

std::string diagonal(std::istream& in) {
    const size_t count = read<size_t>(in);
    std::vector<std::pair<Integer,Vector>> terms;
    Vector axial(3), weights(3);
    for (size_t i = 0; i < count; ++i) {
        Vector powers = row<Integer>(in, 3);
        Integer degree = std::accumulate(powers.begin(), powers.end(), Integer(0));
        for (size_t j = 0; j < 3; ++j)
            if (degree && powers[j] == degree && (!axial[j] || degree < axial[j]))
                axial[j] = degree;
        terms.emplace_back(degree, std::move(powers));
    }
    if (*std::min_element(axial.begin(), axial.end()) < 2) return "none";
    Integer level = lcm(axial[0], lcm(axial[1], axial[2]));
    for (size_t j = 0; j < 3; ++j) {
        weights[j] = level/axial[j];
        if (weights[j] > 1024) return "none";
    }
    for (const auto& [degree, powers] : terms) {
        const auto weight = std::inner_product(powers.begin(), powers.end(), weights.begin(), Integer(0));
        if (weight < level) return "none";
        if (weight == level && std::find(powers.begin(), powers.end(), degree) == powers.end())
            return "none";
    }
    std::sort(terms.rbegin(), terms.rend());
    std::ostringstream out;
    out << level; write(out, weights); out << '\n';
    for (size_t j = 0; j < 3; ++j) {
        Vector powers(3);
        powers[j] = axial[j];
        write(out, powers); out << '\n';
    }
    for (const auto& term : terms) { write(out, term.second); out << '\n'; }
    return out.str();
}

std::string compress(std::istream& in, size_t n) {
    const size_t count = read<size_t>(in);
    Vector strides(n);
    Rays support;
    for (size_t i = 0; i < count; ++i) {
        support.push_back(row<Integer>(in, n));
        for (size_t j = 0; j < n; ++j) strides[j] = gcd(strides[j], support.back()[j]);
    }
    for (auto& stride : strides) if (!stride) stride = 1;
    std::ostringstream out;
    write(out, strides); out << '\n';
    for (Vector& powers : support) {
        for (size_t j = 0; j < n; ++j) powers[j] /= strides[j];
        write(out, powers); out << '\n';
    }
    return out.str();
}

std::string isolated_monomial(std::istream& in, size_t n) {
    const size_t count = read<size_t>(in), rank = read<size_t>(in);
    const Face axes = row<size_t>(in, rank);
    Rays support;
    for (size_t i = 0; i < count; ++i) support.push_back(row<Integer>(in, n));
    for (size_t j : axes) {
        if (j >= n) throw std::invalid_argument("Invalid projection coordinate.");
        std::map<Integer, size_t> counts;
        for (const auto& e : support) {
            ++counts[e[j]];
            if (counts.size() > 2) break;
        }
        if (counts.size() == 2 && (counts.begin()->second == 1 || counts.rbegin()->second == 1))
            return std::to_string(j);
    }
    return "none";
}

std::string excluded_faces(std::istream& in, size_t dimension) {
    const size_t count = read<size_t>(in);
    std::set<Face> candidates;
    for (size_t j = 0; j < count; ++j) {
        Face difference;
        for (size_t i = 0; i < dimension; ++i) if (!read<int>(in)) difference.push_back(i);
        candidates.insert(std::move(difference));
    }
    std::vector<Face> ordered(candidates.begin(), candidates.end()), minimal;
    std::sort(ordered.begin(), ordered.end(), [](const Face& a, const Face& b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b;
    });
    std::ostringstream out;
    for (const auto& face : ordered) {
        if (std::any_of(minimal.begin(), minimal.end(), [&](const Face& previous) {
                return std::includes(face.begin(), face.end(), previous.begin(), previous.end());
        })) continue;
        minimal.push_back(face);
        write(out, face); out << '\n';
    }
    return out.str();
}

std::string axial_data(std::istream& in, size_t n) {
    const size_t count = read<size_t>(in);
    Vector axial(n), expansion(n, count);
    for (size_t k = 0; k < count; ++k) {
        const auto powers = row<Integer>(in, n);
        const auto degree = std::accumulate(powers.begin(), powers.end(), Integer(0));
        for (size_t j = 0; j < n; ++j) {
            expansion[j] += powers[j];
            if (degree && powers[j] == degree && (!axial[j] || degree < axial[j])) axial[j] = degree;
        }
    }
    Integer volume = 1;
    for (const auto& power : axial) if (power) volume *= power;
    std::ostringstream out;
    out << volume; write(out, axial); out << '\n'; write(out, expansion);
    return out.str();
}

std::string coordinate_shear(std::istream& in, size_t n) {
    size_t donor = read<size_t>(in), axis = read<size_t>(in);
    if (axis >= n || donor >= n || axis == donor) throw std::invalid_argument("Invalid shear axes.");
    std::ostringstream out;
    for (size_t i = 0; i < n; ++i) {
        auto image = row<Integer>(in,n);
        image[axis] += image[donor];
        write(out, image); out << '\n';
    }
    return out.str();
}

// Geometric face on every ray and every cone stratum. Intersecting their
// sorted minima is equivalent to taking the initial form on the sum ray.
std::string initial_faces(std::istream& in, size_t n) {
    const size_t nr = read<size_t>(in), nc = read<size_t>(in), nt = read<size_t>(in);
    if (!nt) throw std::invalid_argument("Empty Newton support.");
    std::vector<Face> minima;
    std::ostringstream out;
    for (size_t i = 0; i < nr; ++i) {
        const auto values = row<Integer>(in, nt);
        const Integer minimum = *std::min_element(values.begin(), values.end());
        Face mask;
        for (size_t j = 0; j < nt; ++j) if (values[j] == minimum) mask.push_back(j);
        out << minimum; write(out, mask); out << '\n';
        minima.push_back(std::move(mask));
    }
    std::set<Face> seen;
    for (size_t i = 0; i < nc; ++i) {
        Face cone = sorted(row<size_t>(in, n));
        if (cone.back() >= nr) throw std::invalid_argument("Invalid Newton ray.");
        for (size_t k = 1; k <= n; ++k) subsets(cone, k, [&](const Face& face) {
            if (face.empty() || face.back() < n || !seen.insert(face).second) return;
            Face mask = minima[face.front()];
            for (size_t j = 1; j < k; ++j) {
                Face common;
                const auto& other = minima[face[j]];
                std::set_intersection(mask.begin(), mask.end(), other.begin(), other.end(), std::back_inserter(common));
                mask = std::move(common);
            }
            if (mask.empty()) throw std::invalid_argument("Newton chart lacks a common minimizing monomial.");
            out << k; write(out, face); write(out, mask); out << '\n';
        });
    }
    return out.str();
}

std::string repair_orbits(std::istream& in, size_t n) {
    const auto nr = read<size_t>(in), nc = read<size_t>(in);
    Rays rays;
    for (size_t i = 0; i < nr; ++i) rays.push_back(row<Integer>(in, n));
    std::ostringstream out;
    std::set<Face> seen;
    for (size_t i = 0; i < nc; ++i) {
        auto cone = sorted(row<size_t>(in, n));
        if (cone.back() >= nr) throw std::invalid_argument("Invalid Newton ray.");
        if (cone.back() < n) return "none";
        for (size_t k = 1; k <= n; ++k) subsets(cone, k, [&](const Face& orbit) {
            if (orbit.empty() || orbit.back() < n || !seen.insert(orbit).second) return;
            for (size_t j = 0; j < n; ++j)
                if (std::none_of(orbit.begin(), orbit.end(), [&](size_t r){ return rays[r][j] != 0; }))
                    return;
            const auto d = n-k;
            out << i << ' ' << d << ' ' << (d % 2 ? 1 : -1);
            write(out, orbit); out << '\n';
        });
    }
    return out.str();
}

std::string repair_orders(std::istream& in, size_t n) {
    const size_t nr = read<size_t>(in), parts = read<size_t>(in);
    auto total_orders = row<Integer>(in,nr);
    const auto weights = row<Integer>(in,parts);
    Vector weighted(nr);
    for (const auto& weight : weights) {
        const auto orders = row<Integer>(in,nr);
        for (size_t j = 0; j < nr; ++j) {
            total_orders[j] -= orders[j];
            weighted[j] += weight * orders[j];
        }
    }
    if (std::any_of(total_orders.begin(), total_orders.end(), [](const auto& value){ return value != 0; }))
        return "none";
    std::ostringstream out;
    for (size_t i = 0; i < nr; ++i) {
        const auto ray = row<Integer>(in,n);
        out << weighted[i] << ' ' << std::accumulate(ray.begin(), ray.end(), Integer(0)) << '\n';
    }
    return out.str();
}

std::string origin_orbits(std::istream& in, size_t dimension) {
    if (dimension > 8) throw std::invalid_argument("Unsupported torus chart dimension.");
    std::vector<unsigned> coverage(dimension);
    const unsigned full = (1U << dimension) - 1;
    for (size_t i = 0; i < dimension; ++i) {
        const auto ray = row<Integer>(in, dimension);
        for (size_t j = 0; j < dimension; ++j) if (ray[j]) coverage[i] |= 1U << j;
    }
    Face axes(dimension);
    std::iota(axes.begin(), axes.end(), 0);
    std::ostringstream out;
    for (size_t size = 1; size <= dimension; ++size) subsets(axes, size, [&](const Face& face){
        unsigned image = 0;
        for (auto i : face) image |= coverage[i];
        if (image == full) { write(out, face); out << '\n'; }
    });
    return out.str();
}

std::string terminal_sections(std::istream& in, size_t n) {
    const auto count = read<size_t>(in);
    const auto exceptional_count = read<size_t>(in);
    const auto copies = read<Integer>(in);
    const auto exceptional = row<size_t>(in, exceptional_count);
    const std::set<size_t> selected(exceptional.begin(), exceptional.end());
    Integer total = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto cone = row<size_t>(in, n);
        const std::set<size_t> rays(cone.begin(), cone.end());
        size_t axes = std::count_if(rays.begin(), rays.end(), [&](size_t ray){ return selected.count(ray); });
        total += ((Integer(1) << axes) - 1) * copies;
        if (total > 1000000) return "budget";
    }
    return "ok";
}

std::string separated(std::istream& in, size_t n) {
    const size_t count = read<size_t>(in);
    for (size_t i = 0; i < count; ++i) {
        const auto weights = row<Integer>(in,n);
        if (*std::min_element(weights.begin(),weights.end()) < 0 &&
            *std::max_element(weights.begin(),weights.end()) > 0) return "0";
    }
    return "1";
}

std::string diagonal_cells(std::istream& in) {
    const auto nr = read<size_t>(in), nc = read<size_t>(in);
    const bool intrinsic = read<int>(in) != 0;
    std::vector<Face> faces;
    for (size_t i = 0; i < nr; ++i) {
        size_t count = read<size_t>(in);
        faces.push_back(row<size_t>(in, count));
    }
    std::set<std::pair<size_t,size_t>> pairs;
    std::vector<Face> triangles;
    for (size_t i = 0; i < nc; ++i) {
        auto cone = row<size_t>(in, 3);
        if (*std::max_element(cone.begin(), cone.end()) >= nr)
            throw std::invalid_argument("Invalid diagonal fan cone.");
        if (!intrinsic && *std::min_element(cone.begin(),cone.end()) >= 3)
            triangles.push_back(cone);
        std::sort(cone.begin(), cone.end());
        for (size_t j = 0; j < 2; ++j)
            for (size_t k = j+1; k < 3; ++k)
                if (cone[j] >= 3) pairs.emplace(cone[j], cone[k]);
    }
    std::ostringstream out;
    for (const auto& [a,b] : pairs) {
        Face common;
        if (intrinsic) {
            std::set_intersection(faces[a].begin(), faces[a].end(), faces[b].begin(), faces[b].end(),
                                  std::back_inserter(common));
            if (common.size() != 2) continue;
        }
        out << "edge " << a << ' ' << b; write(out, common); out << '\n';
    }
    for (const auto& cone : triangles) {
        out << "triangle"; write(out,cone); out << '\n';
    }
    return out.str();
}

std::string signed_split(std::istream& in, size_t n) {
    const auto powers = row<Integer>(in, n);
    Vector positive(n), negative(n);
    for (size_t i = 0; i < n; ++i) {
        if (powers[i] < 0) negative[i] = -powers[i];
        else positive[i] = powers[i];
    }
    std::ostringstream out;
    write(out, positive); out << '\n';
    write(out, negative); out << '\n';
    return out.str();
}

std::string incidence(std::istream& in,size_t n) {
    size_t nr=read<size_t>(in), nc=read<size_t>(in), ne=read<size_t>(in);
    bool intrinsic=read<int>(in)!=0;
    Vector delta=row<Integer>(in,n);
    Face exceptional=row<size_t>(in,ne);
    std::set<size_t> selected(exceptional.begin(),exceptional.end());
    Vector signs;
    Rays rays;
    for(size_t i=0;i<nr;++i) {
        rays.push_back(row<Integer>(in,n));
        signs.push_back(std::inner_product(rays.back().begin(),rays.back().end(),delta.begin(),Integer(0)));
    }
    std::map<Piece,Seen,ShortFirst> pieces;
    using Local=std::map<Face,Face>;
    std::vector<Local> charts(nc);
    for(size_t ci=0;ci<nc;++ci) {
        const Face c=row<size_t>(in,n); Face pos,neg,axes;
        for(auto r:c) {
            if(r>=nr) throw std::invalid_argument("Invalid cone ray.");
            if(signs[r]>0) pos.push_back(r);
            if(signs[r]<0) neg.push_back(r);
            if(selected.count(r)) axes.push_back(r);
        }
        std::sort(axes.begin(),axes.end());
        std::vector<bool> covered(n,false);
        for(auto r:c) if(!intrinsic || (!pos.empty()&&!neg.empty()) || signs[r]==0) {
            for(size_t j=0;j<n;++j) if(rays[r][j]!=0) covered[j]=true;
        }
        bool over=std::all_of(covered.begin(),covered.end(),[](bool b){return b;});
        for(size_t k=1;k<=axes.size();++k) subsets(axes,k,[&](const Face& s){
            Face zeros=s;
            if(intrinsic) {
                bool p=false,m=false;
                for(auto r:s){p=p||signs[r]>0;m=m||signs[r]<0;}
                if(p&&m) throw std::invalid_argument("Nonterminal chart: both binomial terms vanish on an exceptional intersection.");
                if(p||m){
                    const auto& live=p?neg:pos;
                    if(live.empty()) return;
                    const auto& vanishing=p?pos:neg;
                    const bool graph=vanishing.size()==1 && abs(signs[vanishing[0]])==1
                                     && live.size()==1 && !selected.count(live[0]);
                    if(live.size()!=1 || (abs(signs[live[0]])!=1 && !graph))
                        throw std::invalid_argument("Nonterminal chart: the surviving monomial is not a single linear coordinate.");
                    zeros.insert(std::lower_bound(zeros.begin(),zeros.end(),live[0]),live[0]);
                }
            }
            charts[ci][s]=zeros;
            auto it=pieces.emplace(Piece{s,zeros},Seen{ci,over}).first;
            it->second.meets |= over;
        });
    }
    std::ostringstream out;
    write(out,signs); out<<"\n";
    std::map<Piece,size_t> ids;
    for(const auto& [piece,data]:pieces) if(data.meets) {
        const auto& s=piece.first; const auto& z=piece.second;
        ids[piece]=ids.size();
        out<<"p "<<s.size(); write(out,s); out<<" "<<z.size(); write(out,z);
        if(s.size()>1) {
            auto parent=[&](const Face& subset){out<<" "<<ids.at({subset,charts[data.witness].at(subset)});};
            for(auto r:s) parent({r});
            subsets(s,s.size()-1,parent);
        }
        out<<"\n";
    }
    for(auto& chart:charts){
        out<<"c";
        for(auto& [s,z]:chart) if(s.size()==1){
            auto it=ids.find({s,z}); if(it!=ids.end()) out<<" "<<s[0]<<" "<<it->second;
        }
        out<<"\n";
    }
    return out.str();
}
}
extern "C" char* atlas_binomial(const char* data,size_t length) noexcept {
    try {
        std::istringstream in(std::string(data,length));
        const auto op=read<std::string>(in); size_t n=read<size_t>(in);
        if(n && op=="newton_edge") return atlas::copy_text(newton_support(in,n,true));
        if(n && op=="newton_project") return atlas::copy_text(newton_support(in,n,false));
        if(n && op=="newton_coefficients") return atlas::copy_text(face_coefficients(in));
        if(n && op=="newton_scaled_degree") {
            const auto stride = read<Integer>(in);
            Integer degree = 0;
            for (size_t i = 0; i < n; ++i) degree = std::max(degree, read<Integer>(in));
            return atlas::copy_text(Integer(degree*stride).str());
        }
        if(op=="exceptional_data") return atlas::copy_text(exceptional_data(in,n));
        if(op=="surface_curve_choice" || op=="surface_curve_matches")
            return atlas::copy_text(surface_curve_choice(in,n,op=="surface_curve_matches"));
        if(n && op=="origin_orbits") return atlas::copy_text(origin_orbits(in,n));
        if(n && op=="newton_repair_orbits") return atlas::copy_text(repair_orbits(in,n));
        if(n && op=="newton_repair_orders") return atlas::copy_text(repair_orders(in,n));
        if(n && op=="newton_separating_bound") {
            auto count = read<Integer>(in);
            return atlas::copy_text(Integer((n-1)*count*(count-1)/2+1).str());
        }
        if(n && op=="newton_separating_coefficients") {
            auto a = read<Integer>(in);
            Integer power = 1;
            std::ostringstream out;
            for (size_t j = 0; j < n; ++j) { out << ' ' << power; power *= a; }
            return atlas::copy_text(out.str());
        }
        if(n && op=="newton_axial") return atlas::copy_text(axial_data(in,n));
        if(n && op=="newton_shear") return atlas::copy_text(coordinate_shear(in,n));
        if(n && op=="newton_faces") return atlas::copy_text(initial_faces(in,n));
        if(n && op=="common_coordinate") {
            const size_t count = read<size_t>(in);
            std::vector<bool> common(n,true);
            for (size_t i = 0; i < count; ++i) {
                auto powers = row<Integer>(in,n);
                for (size_t j = 0; j < n; ++j) if (!powers[j]) common[j] = false;
            }
            return atlas::copy_text(std::any_of(common.begin(), common.end(), [](bool x){return x;}) ? "1" : "0");
        }
        if(n && op=="signed_split") return atlas::copy_text(signed_split(in,n));
        if(n && op=="minima") return atlas::copy_text(minima(in,n));
        if(n && op=="residual_orders") {
            auto values = row<Integer>(in, n);
            const Integer minimum = *std::min_element(values.begin(), values.end());
            for (auto& order : values) order -= minimum;
            std::ostringstream result;
            result << minimum;
            write(result, values);
            return atlas::copy_text(result.str());
        }
        if(n && op=="character") return atlas::copy_text(character(in,n));
        if(n && op=="excluded_faces") return atlas::copy_text(excluded_faces(in,n));
        if(n && op=="newton_compress") return atlas::copy_text(compress(in,n));
        if(n && op=="newton_isolate") return atlas::copy_text(isolated_monomial(in,n));
        if(n && op=="separated") return atlas::copy_text(separated(in,n));
        if(n && op=="terminal_sections") return atlas::copy_text(terminal_sections(in,n));
        if(n==3 && op=="diagonal_cells") return atlas::copy_text(diagonal_cells(in));
        if(n==3 && op=="diagonal") return atlas::copy_text(diagonal(in));
        if(n<2) throw std::invalid_argument("Invalid binomial dimension.");
        if(op=="fan" || op=="intrinsic_fan") return atlas::copy_text(fan(in,n,op=="intrinsic_fan"));
        if(op=="incidence") return atlas::copy_text(incidence(in,n));
        if(op=="pairings") return atlas::copy_text(pairings(in,n));
        throw std::invalid_argument("Unknown binomial operation.");
    } catch(const std::exception& e){atlas::set_error(e.what());}
    catch(...){atlas::set_error("Unknown binomial failure.");}
    return nullptr;
}
extern "C" int atlas_newton_support_abi() noexcept { return 1; }
