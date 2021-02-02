function [J,thetaOne,thetaTwo,thetaThree] = nn_two_hidden_layers(thetaOne,thetaTwo,thetaThree,x,y,numOfIts,alpha)
  

  y=ind2vec(y');
  
  M=size(x,1);
  %Adding Bias
  x = [ones(M,1) x];
  
  for i=1:numOfIts

    i
      
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
  
  
  J=(1/M) * sum ( sum ( (-y) .* log(a4) - (1-y) .* log(1-a4) ));
  
  
  %Backward Propagation
  
 
  
  delta4 = a4 - y;
  
  z3=[ones(1,size(z3,2));z3];
  delta3 = (thetaThree' * delta4) .* compute_sigmoid(z3,1);   
  delta3(1,:) = [];
  
  
  z2=[ones(1,size(z2,2));z2];
  delta2 = (thetaTwo' * delta3) .* compute_sigmoid(z2,1);
  delta2(1,:) = [];
  
  %Theta updation
  thetaThree = thetaThree - ((alpha * delta4 * a3)/M);
  thetaTwo = thetaTwo - ((alpha * delta3 * a2)/M);
  thetaOne = thetaOne - ((alpha * delta2 * x)/M);
 
  end
end
