
clear all

%----- Define Network Parameters ---- %
Nodes=45;   % Number of nodes in hidden layer
ID=2;       % Number of input delays (0:ID)
FD=3;       % Number of feedback delays (1:FD)

for Input_Delay=ID
    for Feedback_Delay=FD
        
        clearvars -except Nodes Feedback_Delay Input_Delay Exp
        close all
        
        %-------- Load Dataset ------%
        File_Name="Mahdiar_Edraki_IntroToML_FinalProject_Dataset.mat";
        load(File_Name)
        %------ END Load Dataset ----%
        
        % Create Training and Testing Dataset called X and T
        T = con2seq(y);
        X = con2seq(u);
        
        % Create Additional Validation Dataset called Xnew and Tnew
        Tnew = con2seq(ynew);
        Xnew = con2seq(unew);
        
        % Train a network, and simulate it using X and T dataset
        net = narxnet(0:Input_Delay,1:Feedback_Delay,Nodes,'open');
        net.trainFcn = 'trainbr';
        net.trainParam.epochs=40;
        [Xs,Xi,Ai,Ts] = preparets(net,X,{},T);
        [net,tr] = train(net,Xs,Ts,Xi,Ai);
        plotperform(tr)
        [Xsnew,Xinew,Ainew,Tsnew] = preparets(net,Xnew,{},Tnew);
        view(net)
        
        %%%----------- Simulate a different Trial - Open Loop --------------%%%
        Delay_Out=Feedback_Delay+Input_Delay-1;
        [Y,Xf,Af] = net(Xsnew,Xinew,Ainew);
        performance_open = perform(net,Tsnew,Y)
        LENGTH=length(ynew(1,:));
        for J=1:LENGTH-Delay_Out
            y1_Corrected(:,J)=Y{1,J};
        end
        
            Max_Y_Lim=nanmax(ynew(:,:),[],'all')/1000;
            Min_Y_Lim=nanmin(ynew(:,:),[],'all')/1000;
            figure
            ROW=[1 11 21];
            for Marker=1:30
                if ismember(Marker,[1:3:30])
                    subplot(3,10,ROW(1))
                    plot((1:LENGTH)/500,ynew(Marker,:)/1000,'linewidth',1,'color','blue')
                    hold on
                    plot((1:LENGTH-Delay_Out)/500,y1_Corrected(Marker,:)/1000,'linewidth',1,'color','red')
                    ylim([Min_Y_Lim Max_Y_Lim])
                    set(gca,'FontSize',16)
                    if Marker==1
                        ylabel("X Position (m)")
                    end
                    ROW(1)=ROW(1)+1;
                elseif ismember(Marker,[2:3:30])
                    subplot(3,10,ROW(2))
                    plot((1:LENGTH)/500,ynew(Marker,:)/1000,'linewidth',1,'color','blue')
                    hold on
                    plot((1:LENGTH-Delay_Out)/500,y1_Corrected(Marker,:)/1000,'linewidth',1,'color','red')
                    ylim([Min_Y_Lim Max_Y_Lim])
                    set(gca,'FontSize',16)
                    if Marker==2
                        ylabel("Y Position (m)")
                    end
                    ROW(2)=ROW(2)+1;
                else
                    subplot(3,10,ROW(3))
                    plot((1:LENGTH)/500,ynew(Marker,:)/1000,'linewidth',1,'color','blue')
                    hold on
                    plot((1:LENGTH-Delay_Out)/500,y1_Corrected(Marker,:)/1000,'linewidth',1,'color','red')
                    ylim([Min_Y_Lim Max_Y_Lim])
                    set(gca,'FontSize',16)
                    if Marker==3
                        ylabel("Z Position (m)")
                    end
                    xlabel("Time (sec)")
                    ROW(3)=ROW(3)+1;
                end
            end
            sgtitle("Performance in Open Loop")
        %%%--------- END Simulate a different Trial - Open Loop --------------%%%
        
        %%%%---------- Train Closed Loop ------%%%%
        clearvars Xs Xic Aic Ts
        [netc,~,~] = closeloop(net,Xinew,Ainew);
        netc.trainParam.epochs=80;
        view(netc)
        [Xs,Xic,Aic,Ts] = preparets(netc,X,{},T);
        [netc,trc] = train(netc,Xs,Ts,Xic,Aic);
        plotperform(trc)
        %%%%-------- END Train Closed Loop ------%%%%
        
        %%%----------- Simulate a different dataset - Closed Loop --------------%%%
        clearvars Xsnew Xic Aic Tsnew
        [Xsnew,Xic,Aic,Tsnew] = preparets(netc,Xnew,{},Tnew);
        y2 = netc(Xsnew,Xic,Aic);
        % % Calculate the network performance.
        [Y2,Xf,Af] = netc(Xsnew,Xic,Aic);
        
        performance_closed = perform(netc,Tsnew,Y2)
        LENGTH=length(ynew(1,:));
        for J=1:LENGTH-Delay_Out
            y3_Corrected(:,J)=Y2{1,J};
        end
        Max_Y_Lim=nanmax(ynew(:,:),[],'all')/1000;
        Min_Y_Lim=nanmin(ynew(:,:),[],'all')/1000;
        figure
        ROW=[1 11 21];
        for Marker=1:30
            if ismember(Marker,[1:3:30])
                subplot(3,10,ROW(1))
                plot((1:LENGTH)/500,ynew(Marker,:)/1000,'linewidth',1,'color','blue')
                hold on
                plot((1:LENGTH-Delay_Out)/500,y3_Corrected(Marker,:)/1000,'linewidth',1,'color','red')
                ylim([Min_Y_Lim Max_Y_Lim])
                set(gca,'FontSize',16)
                if Marker==1
                    ylabel("X Position (m)")
                end
                ROW(1)=ROW(1)+1;
            elseif ismember(Marker,[2:3:30])
                subplot(3,10,ROW(2))
                plot((1:LENGTH)/500,ynew(Marker,:)/1000,'linewidth',1,'color','blue')
                hold on
                plot((1:LENGTH-Delay_Out)/500,y3_Corrected(Marker,:)/1000,'linewidth',1,'color','red')
                ylim([Min_Y_Lim Max_Y_Lim])
                set(gca,'FontSize',16)
                if Marker==2
                    ylabel("Y Position (m)")
                end
                ROW(2)=ROW(2)+1;
            else
                subplot(3,10,ROW(3))
                plot((1:LENGTH)/500,ynew(Marker,:)/1000,'linewidth',1,'color','blue')
                hold on
                plot((1:LENGTH-Delay_Out)/500,y3_Corrected(Marker,:)/1000,'linewidth',1,'color','red')
                ylim([Min_Y_Lim Max_Y_Lim])
                set(gca,'FontSize',16)
                if Marker==3
                    ylabel("Z Position (m)")
                end
                xlabel("Time (sec)")
                ROW(3)=ROW(3)+1;
            end
        end
        sgtitle("Performance in Closed Loop")
        %%%--------- END Simulate a different dataset - Closed Loop --------------%%%
        
        %             File_Name="Workspace_OpenAndClosed_ID"+Input_Delay+"_FD"+Feedback_Delay+"_"+Nodes+".mat";
        %             save(File_Name)
    end
end


