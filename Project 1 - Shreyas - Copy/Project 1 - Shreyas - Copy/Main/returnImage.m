function [B] = returnImage(A)

%splitting all the values in the .btyes file based on spaces
%this provides all couple hexa values and the binary value at the beggning
A = split(A);

%removing the binary values in the beggning of each line
A(1:17:end) = [];

%finding the indices where '??' special character is present
x = find(strcmp(A,'??')==1);

%replacing it as 0 initially to make it pass through hex2dec(), since
%hex2dec does not support neative values
A(x(1:end)) = {'0'};

%converting the entire matrix from hex to dec values
A = hex2dec(A);

%using the already gathered index of '??' and replacing 0's with -1
A(x(1:end)) = -1;


%incases where the number rows is not divisible by 16, we add -1's so that
%the number of raows is divisible by 16
if mod(size(A,1),16) > 0
   
    while mod(size(A,1),16) ~= 0
        A = [A;-1];
    end
   
end

%reshaping the matrix into 16xRows
B = reshape(A,[16 size(A,1)/16])';

%resizing the image into 256X16 using bileniar method
%here bileniar method is used to avoid values beyond the range of gray
%values. The default method being bicubic.

B = imresize(B,[256 16],'bilinear');

end

