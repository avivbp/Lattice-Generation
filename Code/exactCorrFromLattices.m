% path to stored lattice vectors file for which you want to calculate exact correlation matrices
path = "C:\Users\Aviv\PycharmProjects\CLWS2\lattice_vectors_first.mat";
vectors = load(path).lattice_vectors;
matrix_struct = struct();
num_matrices = size(vectors(:,1),1);
mat_array = cell(1, num_matrices);
current_directory = pwd;
disp(current_directory);
%G2;
for k=1:num_matrices
    vect = vectors(k,:);
    H_sp = diag(vect);
    for i = 1:size(H_sp, 1)
            for j = 1:size(H_sp, 2)
                if i == j-1 || j == i-1
                    H_sp(i, j) = 1;
                end
            end
        
    end

    global Fock_basis
    
    
    Nsites=10;      % number of sites
    
    Nparticles=2;   % number of particles
    Gamma=3;        % Interaction as a function of seperation
    
    T=2;      % Times in which the wavefunction is evaluated


    [Fock_basis, ~]=build_Fock_basis(Nsites,Nparticles); % prepare FocK Basis. This is a global variable
    
    %prepare initial sate - in Fock basis
    psi_init=build_psi_init([5 6]);
    
            %H_sp=load ('...');
    
            %H_sp=H_sp.h;
            
            Uij=build_Uij(Nsites,Gamma(1));    % build interaction matrix, can represent long range interactions, or position dependent interactions
            H_BH=build_BH(H_sp,Uij);        % builde the BH many body Hamiltonyan, inclusing interactions
            [V,E]=eig(H_BH);
            
            
            %% propagate according to BH and the intial state
            psi_out=(V*diag(exp(-1i*T*diag(E)))/V)*psi_init;
            
            %% Calculate G2 from the final many body wavefunction
            %tmp=g2_from_psi(psi_out,Nsites);
            %G2 = G2 + tmp;
            G2 = g2_from_psi(psi_out,Nsites);
            field_name = sprintf('matrix%d', k);
            matrix_struct.(field_name) = G2;
            mat_array{k} = G2;
            
            %% save
end
%G2 = G2/num_matrices;
matrix_struct;
% path to save the correlation matrices file
path = "C:\Users\Aviv\PycharmProjects\CLWS2";
save_path = strcat(path,"/corrMatrices_first.mat");
%save(save_path,'G2');
save(save_path,'matrix_struct')

function [n_many_full, n_many]=build_Fock_basis(No_of_sites,N0_of_particles)
    %     H1 - single particle H (even non tridiagonal)
    %     N- number of particles
    %     returnes the many body Hamiltonyan, in the fock basis given by n_many
    
    
    vec=[zeros(1,No_of_sites-1),ones(1,N0_of_particles)];
    index_of_particles=choosenk(N0_of_particles+No_of_sites-1,N0_of_particles);
    D=length(index_of_particles);
    possible_ind=1:D;
    result=zeros(D,N0_of_particles+No_of_sites-1);
    n=zeros(D,N0_of_particles);
    p=N0_of_particles+1;
    for k=1:D
        result(k,index_of_particles(k,:))=ones(1,N0_of_particles); %all states
        tmp_res=[0,result(k,:),0];
        location_of_zeros=find(tmp_res==0);
        for j=1:(length(location_of_zeros)-1);
            n(k,j)=sum(tmp_res(location_of_zeros(j):location_of_zeros(j+1)));
        end
    end
    
    %clean all zero rows
    n(:,find(sum(n)==0))=[];
    %define lookup table
    lookup=sum(n.*(repmat(p.^(0:(No_of_sites-1)),D,1)),2);
    %find matrix elements
    
    
    nn=[''];
    for kk=1:size(n,2);
        nn=[nn num2str(n(:,kk))];
end

n_many=nn;
n_many_full=n;
% figure;
% plot_3d_graph2(gca,U2,0,max(max(U2)),0);
% set(gca,'xtick',[1:size(nn,1)],'xticklabel',nn);
% set(gca,'ytick',[1:size(nn,1)],'yticklabel',nn);
end

function psi_init=build_psi_init(POS);
    % function psi_init=build_psi_init(Pos);
    % POS is a vector giving the position of the particles:
    % psi_init is the initial state in the Fock basis!
    
    global Fock_basis
    
    psi_=zeros(1,size(Fock_basis,2));
    for kk=1:size(POS,2)
        psi_(POS(kk))=psi_(POS(kk))+1;
    end
    
    [~,ind]=ismember(psi_,Fock_basis,'rows');
    
    psi_init_=zeros(size(Fock_basis,1),1);
    psi_init_(ind)=1;
    psi_init=psi_init_;
end

function f=build_Uij(N, U_of_x)
    % function f=createUij(N, U_of_x)
    % builds long range interaction matrix
    % N is number of sites,
    % U_of_x is a vector of interactions U as a funstion of distance
    
    
    a=zeros(N,N);
    a=a+diag(U_of_x(1)*ones(N,1),0);
    
    for kk=2:size(U_of_x,2)
        a=a+diag(U_of_x(kk)*ones(N-kk+1,1),kk-1)+diag(U_of_x(kk)*ones(N-kk+1,1),-kk+1);
    end
    
    f=a;
    
end


function H_many_interacting=build_BH(H1,Uij)
    % function H_many_interacting=build_BH(H1,Fock_basis,Uij)
    % H1 is the single particle H
    % Uij is a matrix representing long range interactions
    global Fock_basis
    
    non_interacting_H_many=build_BH_noninteracting(H1);
    Hint=build_Hint(Uij);
    
    H_many_interacting=(non_interacting_H_many+Hint);
end

function out=g2_from_psi(psi,I)
    %function out=calc_2body_corr(psi,I);
    %this finction calculates the 2 body corelation matrix
    %out=C_ij(t) = <psi(t)| n_i n_j |psi(t)>
    global Fock_basis
    
    sz=size(psi);
    D=sz(1);
    for ind=1:I
        eval(['n',num2str(ind),'=sparse(diag(Fock_basis(:,ind)));']);
    end
    C=zeros(I);
    for j=1:I
        for k=1:I
            eval(['njnk=n',num2str(j),'*n',num2str(k),';']);
            eval(['nj=n',num2str(j),';'])
            eval(['nk=n',num2str(k),';'])
            %         C(j,k)=(psi')*njnk*(psi)-((psi')*nj*(psi))*((psi')*nk*(psi));
            C(j,k)=(psi')*njnk*(psi);
            %         C(j,k)=C(j,k)-0.5*((psi')*nj*(psi))*((psi')*nk*(psi));
            if j==k;
                C(j,k)=(C(j,k))-psi'*nj*psi;
            end
        end
    end
    out=C;
    %
end

function Hint=build_Hint(Uij)
    % Uij is the interaction matrix, can be long range interations!
    % Fock_basis is many body fock basis
    %function Hint=CreateInteractionHamiltonian(Uij,n)
    global Fock_basis
    
    sz=size(Fock_basis);
    D=max(sz);
    I=min(sz);
    Hint=zeros(D);
    for i=1:I
        for j=1:I
            for k=1:D
                if i~=j
                    Hint(k,k)= Hint(k,k)+Uij(i,j).*Fock_basis(k,i)*Fock_basis(k,j);
                else
                    Hint(k,k)= Hint(k,k)+Uij(i,j).*Fock_basis(k,i)*(Fock_basis(k,i)-1)/2;
                end
            end
        end
    end
    
end

function Hmany=build_BH_noninteracting(H1)
    %     H1 - single particle H (even non tridiagonal)
    %     N- number of particles
    %     returnes the many body Hamiltonyan, in the fock basis given by n_many
    global Fock_basis
    
    n=Fock_basis;
    I=size(H1,1); %number of sites
    N=max(max(Fock_basis)); %number of particles
    teta=0;
    
    W_=diag(H1,0)';
    J=1;
    
    % index_of_particles=choosenk(N+I-1,N);
    % D=length(index_of_particles);
    D=size(Fock_basis,1);
    p=N+1;
    
    %define lookup table
    lookup=sum(n.*(repmat(p.^(0:(I-1)),D,1)),2);
    
    %find matrix elements
    M=sparse(D,D);
    M_=M;
    
    for dd=1:I % to make next-nearest neighbot make this 2 or I - but check!!!
        J_offdiag=[zeros(1,dd) diag(H1,dd)'];
        for i=1:I-dd
            step=(i<I)*(i+dd)+(i==I);
            n_new1=n;
            n_new1(:,i)=n(:,i)-1;
            n_new1(:,step)=n(:,step)+1;
            n_new2=n;
            n_new2(:,i)=n(:,i)+1;
            n_new2(:,step)=n(:,step)-1;
            alpha1=n_new1*(p.^(0:I-1)');
            alpha2=n_new2*(p.^(0:I-1)');
            [tmp1,line_index1]=ismember(alpha1,lookup);
            [tmp2,line_index2]=ismember(alpha2,lookup);
            tmp1=find(tmp1);
            tmp2=find(tmp2);
            line_index1=line_index1(tmp1);
            line_index2=line_index2(tmp2);
            %chk1(i)=sum(sum(abs(line_index2-tmp1)));
            %chk2(i)=sum(sum(abs(line_index1-tmp2)));
            M(tmp1+(line_index1-1)*D)=J_offdiag(step)*sqrt(n(tmp1,i).*(n(tmp1,step)+1)).*exp(sqrt(-1)*teta/I);
            M(tmp2+(line_index2-1)*D)=J_offdiag(step)*sqrt(n(tmp2,step).*(n(tmp2,i)+1)).*exp(-sqrt(-1)*teta/I);
            %             I-i;
        end
        
    end
    
    W=repmat(W_,D,1);
    %         H=-J*M+Hint;
    %         H=full((H+H')/2);
    Vext=spdiags(sum(W.*n,2),0,D,D);
    H=J*M+Vext;
    H=(H+H')/2;
    
    Hmany=H;
    
end

function x=choosenk(n,k)
%CHOOSENK All choices of K elements taken from 1:N [X]=(N,K)
% The output X is a matrix of size (N!/(K!*(N-K)!),K) where each row
% contains a choice of K elements taken from 1:N without duplications.
% The rows of X are in lexically sorted order.
%
% To choose from the elements of an arbitrary vector V use
% V(CHOOSENK(LENGTH(V),K)).

% CHOOSENK(N,K) is the same as the MATLAB5 function NCHOOSEK(1:N,K) but is
% much faster for large N and most values of K.

%   Copyright (c) 1998 Mike Brookes,  mike.brookes@ic.ac.uk
%
%      Last modified Thu May 21 18:27:50 1998
%
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kk=min(k,n-k);
if kk<2
   if kk<1
      if k==n
         x=1:n;
      else
         x=[];
      end
   else
      if k==1
         x=(1:n)';
      else
         x=1:n;
         x=reshape(x(ones(n-1,1),:),n,n-1);
      end
   end   
else
   n1=n+1;
   m=prod(n1-kk:n)/prod(1:kk);
   x=zeros(m,k);
   f=n1-k;
   x(1:f,k)=(k:n)';
   for a=k-1:-1:1
      d=f;
      h=f;
      x(1:f,a)=a;
      for b=a+1:a+n-k
         d=d*(n1+a-b-k)/(n1-b);
         e=f+1;
         f=e+d-1;
         x(e:f,a)=b;
         x(e:f,a+1:k)=x(h-d+1:h,a+1:k);
      end
   end
end
end
