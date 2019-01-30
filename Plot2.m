cell = Perceptron;

Iteration = 2000;
LearningRate = 0.00949;
Inputs = [-0.329,1; -0.71,-0.2; 0.45,1.75; 0.32,0.44; -0.3,0.85;
          1.56,1.81; 1.7,2; 1.33,1.88; 1.9,1.22; 1.91,1.9;
          0.929,1.23; 0.79,0.33; -0.335,-0.945; 0.62,0.74; -0.7,0.9;
          1.56,-0.11; -0.7,1.92; -0.73,1.74; 1.07,-0.77; -1.91,0.9];
Labels = [1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];

cell = cell.setInfo(Inputs, Labels, LearningRate, Iteration);
cell = cell.Learning();
%disp(cell.w1_revise);
figure(1);
    plot(Inputs(1:10, 1),Inputs(1:10, 2), 'ro');
    hold on
    plot(Inputs(11:20, 1), Inputs(11:20, 2),'bo');
    hold on
    x = linspace(-2, 6, 140);
    y = ((-x*cell.w(1))/cell.w(2))+cell.b/cell.w(2);
    plot(x, y);
    ylim([-2,5.5])
    legend('Class 1', 'Class 2', 'Hyper Plain');
    title('Incorrect-Partition of Scatters');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Nonlinear1.png');
    hold off
   
figure(2);
    plot(cell.w1_revise(1:300),'r-','LineWidth',1.3);
    hold on
    plot(cell.w2_revise(1:300), 'k-','LineWidth',1.3);
    hold on
    plot(cell.b_revise(1:300), 'b-','LineWidth',0.8);
    legend('w1', 'w2', 'b');
    title('Non-Convergence of Parameters');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\NonConvergence1.png');
    hold on
    ylim([-0.6 0.8]);
    hold off