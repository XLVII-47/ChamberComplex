using TimerOutputs
using LinearAlgebra
using AbstractAlgebra
using Combinatorics
using Oscar

function find_Is(subset,m)
    Im = Matrix{Int64}(I, m, m)
    for s in subset
      Im[s,s]=0.0;
    end
  
    return mapreduce(transpose, vcat,[Im[:,x] for x in 1:m ])
end 

function CheckR(A,R)
    for r in R
      if Oscar.dim(Oscar.Polyhedron(A,r[:])) <= 0
        return false
      end
    end
    return true
end

function ps(x::Vector{T}) where T
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    result
end


function ChamberComplex(A)
    m,d=size(A)
    if rank(A)==d
  
      #println(m,d," ", rank(Matrix))
      #S = subsets(1:m,d)
      S = collect(powerset(1:m,d,d))
      #println(S)
      Id = Matrix{Int64}(I, m, m); 
      
      Acolums =mapreduce(transpose, vcat ,[(A[:,y]) for y in 1:d])
      #zerosm =mapreduce(transpose, vcat , 0*[1:m])
      zerosm = zeros((1, m))
      
      Cones = []
     # println(Acolums)
    
      for s in S
        Is=find_Is(s,m)
        CAs=  convex_hull(zerosm,Is,Acolums)
        push!(Cones,CAs)
      end
      
     
      Pcones= ps(Cones)
      
      #FİND RIGHT KERNEL 
      S = MatrixSpace(AbstractAlgebra.ZZ, d, m)
      dd,RK = right_kernel(S(transpose(A)))
      RK = transpose(RK)
      GRK = Array(RK)
      GRK = mapreduce(transpose, vcat ,[GRK[i,:] for i in 1:dd])
      RightKernel = convex_hull(zerosm,nothing,GRK)
      
      CC =[]
      for d in Pcones   
        if length(d)!=0
            if d != Cones
                  K=d[1]
                  for p in d[2:end]
                    K= intersect(p,K) 
                  end
                  
                  KK = intersect(K,RightKernel)
                
                  R= Oscar.rays(K)

                  if CheckR(A,R)
                    push!(CC,KK) 
                  end  
            end
        end
      end
      
     
      len= length(CC)
      for i in 1:length(CC)
        if CC[i]==0
            continue
        end
        for j in 1:length(CC)
            if i==j || CC[j]==0
                continue
            end
          #println(intersect(CC[i],CC[i])==CC[i])
          if Oscar.dim(CC[i])== Oscar.dim(CC[j]) && intersect(CC[i],CC[j])==CC[i]
            CC[j]=0
          end
        end
      end


    else
      printf("matrix is not full rank")
    end

 
  filter!(x -> x != 0 ,CC)
 return CC
end




function last(Cl)
  ll=length(Cl)

  for i in 1:ll
    if Cl[i] == 0
      continue
    end

    for j in 1:ll
      if i==j || Cl[j]==0
        continue
      end
      if Oscar.intersect(Cl[i],Cl[j]) == Cl[i]
        Cl[j] = 0
      end
    end
  end
  return filter(x -> x != 0 ,Cl)
end


function helper(L,first,last,Output,inter,Rk,A,ChamberComplex)
  if first==last+1
    return
  end
  if length(Output)+1 == last
    return
  end
  for i in first:last        

    if inter != 0
        K= Oscar.intersect(inter,L[i])
        KK= Oscar.intersect(K,Rk)
        if Oscar.dim(KK)==0
          continue
        end
        if CheckR(A,Oscar.rays(K))
          try
            push!(ChamberComplex[dim(KK)],KK)
          catch
            ChamberComplex[dim(KK)]=Any[KK]
          end
        end
        
    else
      K=L[i]
      KK= Oscar.intersect(K,Rk)
      if Oscar.dim(KK)==0
        continue
      end
      if CheckR(A,Oscar.rays(K))
        try
          push!(ChamberComplex[dim(KK)],KK)
        catch
          ChamberComplex[dim(KK)]=Any[KK]
        end 
      end
    end
    
    helper(L,i+1,last,[Output;L[i]],K,Rk,A,ChamberComplex)

  end
end

  
function powerset_f(L,Rk,A,ChamberComplex)
    helper(L,1,length(L),[],0,Rk,A,ChamberComplex)
end




function ChamberComplex_f(A)
  m,d=size(A)
  if rank(A)==d
    S = collect(powerset(1:m,d,d))
    Id = Matrix{Int64}(I, m, m); 
    Acolums =mapreduce(transpose, vcat ,[(A[:,y]) for y in 1:d])
    zerosm = zeros((1, m))
    
    Cones = []
    for s in S
      Is=find_Is(s,m)
      CAs=  convex_hull(zerosm,Is,Acolums)
      push!(Cones,CAs)
    end
    
    #FİND RIGHT KERNEL 
    S = MatrixSpace(AbstractAlgebra.ZZ, d, m)
    dd,RK = right_kernel(S(transpose(A)))
    RK = transpose(RK)
    GRK = Array(RK)
    GRK = mapreduce(transpose, vcat ,[GRK[i,:] for i in 1:dd])
    
    RightKernel = convex_hull(zerosm,nothing,GRK)
    
    CC =Dict()
    
    powerset_f(Cones,RightKernel,A,CC)
    
    ret = Any[]
    for item in CC
      ret= [ret;last(item.second)]
    end
    
    return ret
  else
    printf("matrix is not full rank")
  end
end


A = [1 0 ; 0 1; -1 0; 0 -1; 1 1]

#A=[93 -78 -64 -34; 41 -12 -15 -15; 98 38 -97 46; -58 75 18 54; -66 70 37 -59; 4 61 58 -28]
#A = [1 0 0 ; -1 0 0 ;0 1 0 ; 0 -1 0 ; 0 0 -1;1 1 1]

#A = [3 2 ; 1 11; -13 27; 41 91]


@time k = ChamberComplex_f(A);

