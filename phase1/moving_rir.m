%
% Generate directional audio and save it without Normalization (e.g., from -1 to 1).
%

%%
clear, clc, close all

%%
folder = 'C:/Users/Jana_local/OneDrive - University of Ottawa/RIR-DL/v05/phase1/output_dataset/train';
all_rooms = dir(folder);
tic
for ii = 1:length(all_rooms)
    f_room = all_rooms(ii).name;
    
    % Handle special case first
    if contains(f_room, '.')        % assume no . in folder name
        continue
    end
    
    % This is valid, so use it ...
    all_files = dir( sprintf("%s/%s", folder,f_room) );
    for jj = 1:length(all_files)
        fname = all_files(jj).name;
    
        % Handle special case first
        if ~contains(fname, '.mat')        % assume no . in folder name
            continue
        end
        
        % Read configs
        f_config = sprintf("%s/%s/%s", folder, f_room, fname);
        load(f_config)
        
        %% Apply RIR
        % 1. clean wav
        root = 'C:/Users/Jana_local/OneDrive - University of Ottawa/RIR-DL/v05';
        [in, fs] = audioread(fullfile(root, clean_wav));
        in = in' / max(abs(in(:)));
                
        % 2. Rec Path
        center = mic.rel_position .* double(room.dim);
        rp_path(:,:,1) = repmat([  ...
                   center(1)-mic.distance, center(2), center(3);
                   ], length(in), 1 ...    % x,y,z
                );
        rp_path(:,:,2) = repmat([  ...
                   center(1)+mic.distance, center(2), center(3);
                   ], length(in), 1 ...    
                );
            
            %rp_path(:,:,3) = repmat([  ...
             %      center(1)-2*mic.distance, center(2), center(3);
              %     ], length(in), 1 ...    
               % );
            
            %rp_path(:,:,4) = repmat([  ...
             %      center(1)+2*mic.distance, center(2), center(3);
              %     ], length(in), 1 ...    
               % );
            
        % 3. Speaker path
        [x,y,z] = sph2cart(speaker.phi * pi/180, ...
                            ones(length(in),1) * double(speaker.theta), ...
                            ones(length(in),1)* double(speaker.rho));
        sp_path = repmat(center, length(in),1) + [x, y, z];

        % RIR
        [out,beta_hat] = signal_generator(...
                            double(in), ...
                            double(room.C), ...
                            double(fs), ...
                            double(rp_path), ...
                            double(sp_path), ...
                            double(room.dim), ...
                            double(room.rt60), ...
                            double(room.len), ...
                            'o', ...
                            double(room.order)...
                        );


        % Save
        f_output = strrep(f_config, '.mat', '.wav');
        audiowrite( f_output, out' , fs);

    end
    
end

toc