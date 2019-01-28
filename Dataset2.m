Inputs = [-0.329,1; -0.71,-0.2; 0.45,1.75; 0.32,0.44; -0.3,0.85;
          1.56,1.81; 1.7,2; 1.33,1.88; 1.9,1.22; 1.91,1.9;
          0.929,1.23; 0.79,0.33; -0.335,-0.945; 0.62,0.74; -0.7,0.9;
          1.56,-0.11; -0.7,1.92; -0.73,1.74; 1.07,-0.77; -1.91,0.9];
figure(1);
    plot(Inputs(1:10, 1),Inputs(1:10, 2), 'ro');
    hold on
    plot(Inputs(11:20, 1), Inputs(11:20, 2),'bo');
    hold on
    xlim([-1.5 2.5])
    ylim([-1.5 3.5])
    legend('Class 1', 'Class 2');
    title('Binary Dataset');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Dataset2.png');
    hold off