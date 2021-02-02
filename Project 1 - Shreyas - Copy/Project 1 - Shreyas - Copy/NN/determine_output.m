function [predictedClass] = determine_output(thetaOne, thetaTwo,thetaThree,x)


  M=size(x,1);
  %Adding Bias
  x = [ones(M,1) x];
  % FOrward Propagation
  z2 = thetaOne * x';  
  a2 = compute_sigmoid(z2,0);
  % adding activation node
  a2 = [ones(1,size(a2,2));a2]';
 
  
  z3 = thetaTwo * a2';  
  a3 = compute_sigmoid(z3,0);
  % adding activation node
  a3 = [ones(1,size(a3,2));a3]';
  
  z4 = thetaThree * a3';  
  a4 = compute_sigmoid(z4,0);
 
 
  
 [~,predictedClass] = max(a4);
        
  

end
