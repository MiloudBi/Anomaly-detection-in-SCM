

def generate_signature_matrix_node():
    import numpy as np
    data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)
    sensor_n = data.shape[0]
    #data  = np.array(pd.read_csv(raw_data_path, header = None))[:,2:-1]

    # min-max normalization
    max_value = np.max(data, axis=1)
    min_value = np.min(data, axis=1)
    data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)

    # std normalization
    # data = np.nan_to_num(data)
    # data_mean = np.mean(data, axis = 0)
    # data_std = np.std(data, axis = 0)
    # data = np.transpose(data) - data_mean
    # data = data / (data_std + 1e-5)
    
    data = np.transpose(data)

    # plt.plot(data[3,:])
    # plt.show()

    #multi-scale signature matix generation
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        print ("generating signature with window " + str(win) + "...")
        for t in range(min_time, max_time, gap_time):
            #print t
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        #if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
                        matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            matrix_all.append(matrix_t)
            # if t == 70:
            # 	print matrix_all[6][0]

        path_temp = matrix_data_path + "matrix_win_" + str(win)
        #print np.shape(matrix_all[0])

        np.save(path_temp, matrix_all)
        del matrix_all[:]

    print ("matrix generation finish!")