 It is the MATLAB implementation of the Unified Pin-SVM QPP which can minimize the  Pinball loss function in true sense without bothering about the sign of its parameter t.  All experimental results described in the paper titled ‘Comment on pinball loss support vector machine’ can be obtained via these codes.  

Following are the steps to obtain all numerical results detailed in the paper titled ‘Comment on pinball loss support vector machine’ with single click.
    
1.	Open  the file named ‘main_2linear.m’ in MATLAB. 
2.	For the linear kernel results set kernel=1 at line 277, For non-linear kernel results ,set kernel =2.
3.	Run the code . The numerical results for 20 common benchmark datasets, which have also been detailed  in the paper titled  ‘Comment on pinball loss support vector machine’  will be  briefly  displayed  on command window .
        It can also stored in a excel file ‘nonlin_res_pin.xlsx’ . All of the plots used in the  paper will be generated on by one. 

File Details
The   ‘Unfied_pin_svm.m ‘ is the quadprog implementation of the proposed Unified Pin-SVM model.
The ‘pin_svm.m’ is the quadprog implementation of existing Pin-SVM model.



For any support or queries, contact ltpritamanand@gmail.com