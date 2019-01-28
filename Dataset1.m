Inputs = [-0.329,1; -0.71,-0.2; 0.45,1.75; 0.32,0.44; -0.3,0.85;
          1.56,1.81; 1.7,2; 1.33,1.88; 1.9,1.22; 1.91,1.9];
figure(1);
    plot(Inputs(1:5, 1),Inputs(1:5, 2), 'ro');
    hold on
    plot(Inputs(6:10, 1), Inputs(6:10, 2),'bo');
    hold on
    xlim([-1.5 2.5])
    ylim([-1.5 3.5])
    legend('Class 1', 'Class 2');
    title('Binary Dataset');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\assignments\Dataset1.png');
    hold off