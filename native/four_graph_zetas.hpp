// Fibre integration on the contact tree of four distinct polynomial sections.
// The base is an embedded plane resolution of their reduced contact divisor.
#pragma once
#include "three_graph_zetas.hpp"
#include <array>
#include <set>

namespace atlas {
namespace four_graph_integration {
using three_graph_integration::Integral;
using three_graph_integration::J;
using three_graph_integration::Labels;
using three_graph_integration::Z;

using Cluster = std::vector<int>;
using Weights = std::array<int,4>;
constexpr std::array<std::pair<int,int>,6> pairs{{{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}}};

struct Boundary {
    std::string id;
    Z nu, chi;
    std::array<Z,6> orders{};

    const Z& contact(int a, int b) const {
        if (b<a) std::swap(a,b);
        auto it=std::find(pairs.begin(),pairs.end(),std::make_pair(a,b));
        if (it==pairs.end()) throw std::invalid_argument("Expected two distinct roots.");
        return orders[it-pairs.begin()];
    }
};

inline int weight(const Cluster& cluster, const Weights& weights) {
    int result=0;
    for (int index:cluster) result+=weights[index];
    return result;
}

inline std::string ball(int m) {
    return m==1 ? "strict" : std::to_string(m)+"-root ball";
}

struct Tree {
    Integral& integral;
    std::set<std::string>& recorded;
    std::vector<const Boundary*> axes;
    Cluster roots;
    Weights weights;
    Z chi;
    std::string monodromy_locus; // nonempty only on a transverse plane slice
    bool topological;

    // The contact tree on an open cone is independent of the chosen base arc
    // when its first split in each cluster realizes the coordinatewise least
    // contact monomial. Otherwise these summands cannot be certified.
    void visit(const Cluster& cluster) {
        if (cluster.size()<2) return;
        std::vector<Z> depths;
        for (const auto* axis:axes) {
            Z d=axis->contact(cluster[0],cluster[1]);
            for (size_t i=0;i<cluster.size();++i)
                for (size_t j=i+1;j<cluster.size();++j)
                    d=std::min(d,axis->contact(cluster[i],cluster[j]));
            depths.push_back(std::move(d));
        }
        auto deeper=[&](int a,int b) {
            for (size_t k=0;k<axes.size();++k)
                if (axes[k]->contact(a,b)>depths[k]) return true;
            return false;
        };
        std::vector<Cluster> children;
        for (int root:cluster) {
            auto group=std::find_if(children.begin(),children.end(),[&](const Cluster& c) {
                return deeper(root,c.front());
            });
            if (group==children.end()) children.push_back({root});
            else group->push_back(root);
        }
        if (children.size()<2) throw std::invalid_argument("Incomparable contact vectors.");
        for (size_t i=0;i<children.size();++i) for (size_t j=i;j<children.size();++j)
            for (int a:children[i]) for (int b:children[j])
                if (a!=b && deeper(a,b)!=(i==j))
                    throw std::invalid_argument("The contacts do not define a split on this cone.");

        Labels denominators;
        const Z split_chi=(1-static_cast<int>(children.size()))*chi;
        for (size_t k=0;k<axes.size();++k) {
            const auto& axis=*axes[k];
            const Z& d=depths[k];
            Z n=0;
            for (int root:roots) {
                const Z height=(root==cluster.front()) ? d : std::min(d,axis.contact(root,cluster.front()));
                n+=weights[root]*height;
            }
            const Z nu=axis.nu+d;
            if (topological && n!=0) {
                const std::string label=axis.id+" ("+n.get_str()+","+nu.get_str()+")";
                if (recorded.insert(label).second) integral.factor(label,n,nu);
                denominators.push_back(label);
            } else if (topological && nu!=1) {
                throw std::invalid_argument("Unexpected non-unit constant contact factor.");
            }
            if (!monodromy_locus.empty())
                integral.lefschetz(monodromy_locus+" at contact "+d.get_str(),n,split_chi);
        }
        if (topological) {
            integral.term(denominators,split_chi);
            integral.term(denominators,-chi,ball(weight(cluster,weights)).c_str());
            for (const auto& c:children)
                integral.term(denominators,chi,ball(weight(c,weights)).c_str());
        }
        for (const auto& c:children) visit(c);
    }
    void run() { visit(roots); }
};
} // namespace four_graph_integration

inline Json::Value four_graph_zetas(const Json::Value& base) {
    using namespace four_graph_integration;
    using four_graph_integration::Z;
    Integral integral;
    std::set<std::string> recorded;
    for (int m=1;m<=4;++m) integral.factor(ball(m),m,1,m>1);
    integral.term({ball(4)},1);
    std::map<std::string,Boundary> exceptional;
    const Cluster all{0,1,2,3};
    const Weights simple{1,1,1,1};
    for (const auto& row:base["divisors"]) {
        Boundary axis{row["id"].asString(),Z(row["nu"].asString(),10),Z(row["chi"].asString(),10)};
        if (row["orders"].size()!=6) throw std::invalid_argument("Six pairwise contact orders are required.");
        for (size_t j=0;j<pairs.size();++j) {
            axis.orders[j]=Z(row["orders"][Json::ArrayIndex(j)].asString(),10);
            if (axis.orders[j]<=0) throw std::invalid_argument("Each exceptional contact must be positive.");
        }
        exceptional.emplace(axis.id,axis);
        // The initial blowup of a transverse four-section plane germ has
        // multiplicity four. Each projected base blowup also adds one
        // four-cycle of exceptional lines on the nonzero Milnor fibres.
        integral.lefschetz(axis.id+" open, initial",4,axis.chi);
        integral.lefschetz(axis.id+" base blowup correction",4,-1);
        Tree{integral,recorded,{&axis},all,simple,axis.chi,axis.id,true}.run();
    }
    for (const auto& edge:base["edges"]) {
        const auto& a=exceptional.at(edge[0].asString());
        const auto& b=exceptional.at(edge[1].asString());
        Tree{integral,recorded,{&a,&b},all,simple,1,"",true}.run();
        // The least contact monomial contains both base coordinates, so
        // the quasi-ordinary crossing has alternating zeta 1-t^4.
        integral.lefschetz(a.id+" / "+b.id,4,1);
    }
    for (const auto& branch:base["branches"]) {
        const auto& exceptional_axis=exceptional.at(branch["divisor"].asString());
        int p=branch["pair"].asInt();
        if (p<0 || size_t(p)>=pairs.size()) throw std::invalid_argument("Unknown pair of graph roots.");
        Boundary strict{"strict contact "+std::to_string(p),1,1};
        strict.orders[p]=1;
        Tree{integral,recorded,{&exceptional_axis,&strict},all,simple,1,"",true}.run();
        // Restricting to this contact branch merges exactly its two roots.
        // Resolve the resulting weighted plane multigerm, retaining the
        // multiplicity two in every ancestral splitting multiplicity.
        auto [a,b]=pairs[p];
        Cluster distinct;
        for (int i:all) {
            if (i==b) continue;
            if (i!=a && exceptional_axis.contact(a,i)!=exceptional_axis.contact(b,i))
                throw std::invalid_argument("Nonmatching contacts on the restricted pair.");
            distinct.push_back(i);
        }
        Weights weights=simple;
        weights[a]=2;
        weights[b]=0;
        auto locus=exceptional_axis.id+" / "+strict.id;
        integral.lefschetz(locus+" initial",4,1);
        Tree{integral,recorded,{&exceptional_axis},distinct,weights,1,locus,false}.run();
    }
    return integral.finish();
}
} // namespace atlas
