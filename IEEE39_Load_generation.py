import andes
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import scipy.sparse
# 设置日志级别，减少不必要的输出
andes.config_logger(stream_level=30)

def save_topology_adjacency(ss, output_dir="dataset"):
    """
    提取并保存系统的拓扑邻接矩阵。
    
    保存两个文件：
    1. adjacency_matrix.npz: 稀疏矩阵文件 (只存储连接关系 0/1)
    2. bus_id_mapping.csv:   记录矩阵索引与实际母线 ID 的对应关系
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 建立索引映射 (Mapping)
    # ss.Bus.idx.v 是母线的实际 ID (如 1, 2, 39)
    # enumerate 生成内部索引 (0, 1, 2...)
    bus_ids = ss.Bus.idx.v
    n_bus = len(bus_ids)
    
    # 字典: {母线实际ID: 矩阵行号}
    bus_map = {uid: i for i, uid in enumerate(bus_ids)}
    
    # 2. 初始化邻接矩阵
    # 使用 Scipy 稀疏矩阵构建 (LIL 格式方便构建，最后转 CSR)
    # 也可以直接用 np.zeros((n_bus, n_bus)) 构建密集矩阵，如果节点数 < 2000
    adj_mat = scipy.sparse.lil_matrix((n_bus, n_bus), dtype=int)
    
    # 3. 遍历线路 (Line) 填充矩阵
    # 在 ANDES 中，传输线和变压器都属于 Line 组
    n_lines = ss.Line.n
    
    # 获取线路两端的母线 ID
    bus1_list = ss.Line.bus1.v
    bus2_list = ss.Line.bus2.v
    u_list = ss.Line.u.v # 线路状态 (1=投入, 0=断开)
    
    count_edges = 0
    
    for i in range(n_lines):
        # 仅当线路投入运行时才视为连接
        if u_list[i] == 1:
            uid1 = bus1_list[i]
            uid2 = bus2_list[i]
            
            # 找到对应的矩阵索引
            if uid1 in bus_map and uid2 in bus_map:
                idx1 = bus_map[uid1]
                idx2 = bus_map[uid2]
                
                # 填充邻接矩阵 (无向图，双向置 1)
                adj_mat[idx1, idx2] = 1
                adj_mat[idx2, idx1] = 1
                count_edges += 1
    
    # 4. 处理对角线 (自环)
    # 对于 GNN (图神经网络) 或 PCMCI，通常需要对角线为 1
    adj_mat.setdiag(1)
    
    # 转为 CSR 格式以便高效存储和计算
    adj_sparse = adj_mat.tocsr()
    adj_dense = adj_mat.toarray() # 密集矩阵形式
    
    # 5. 保存文件
    
    # 方式 A: 保存为 .npz (推荐，体积小，标准格式)
    save_path_sparse = os.path.join(output_dir, "adjacency_sparse.npz")
    scipy.sparse.save_npz(save_path_sparse, adj_sparse)
    
    # 方式 B: 保存为 .npy (密集矩阵，方便直接 load 进 PyTorch/TensorFlow)
    save_path_dense = os.path.join(output_dir, "adjacency_dense.npy")
    np.save(save_path_dense, adj_dense)
    
    # 方式 C: 保存映射关系 (至关重要！)
    mapping_df = pd.DataFrame({
        "Matrix_Index": range(n_bus),
        "Bus_ID": bus_ids
    })
    save_path_map = os.path.join(output_dir, "bus_id_mapping.csv")
    mapping_df.to_csv(save_path_map, index=False)
    
    print(f"[INFO] 拓扑保存成功!")
    print(f"  - 节点数: {n_bus}")
    print(f"  - 边数: {count_edges} (不含自环)")
    print(f"  - 稀疏矩阵: {save_path_sparse}")
    print(f"  - 密集矩阵: {save_path_dense}")
    print(f"  - 索引映射: {save_path_map}")
    
    return adj_dense, mapping_df

def analyze_stability(ss, dura_time = 0.1, threshold_angle=180, threshold_volt_dev=0.2, threshold_freq_dev=0.6):
    """
    分析仿真结果并判定稳定性
    :param ss: ANDES 系统对象
    :param threshold_angle: 功角差阈值 (度)
    :param threshold_volt_dev: 电压偏差阈值 (p.u.)
    :param threshold_freq_dev: 频率最大偏差 (Hz)
    :return: 包含稳定性状态的字典
    """
    
    # 初始化结果字典，默认时间为 None (表示未发生) 
    results = {
        "Status": "Stable", 
        "Reason": [], 
        "Details": [],
        "Time_Angle_Unstable": None,
        "Time_Freq_Unstable": None,
        "Time_Volt_Unstable": None
    }

    if not ss.TDS.converged:
        results["Status"] = "Unstable"
        results["Reason"].append("Diverged")
        results["Details"].append("Simulation Failed")
        # 如果发散，通常认为最后时刻即为崩溃时刻
        t_end = ss.dae.ts.t[-1]
        results["Time_Angle_Unstable"] = t_end
        results["Time_Volt_Unstable"] = t_end
        return results

    # 获取时间轴
    t = np.array(ss.dae.ts.t)    
    t_clear = 5.0 + dura_time
    t_check_indices = np.where(t > t_clear)[0] # 取故障清除后的
    
    if len(t_check_indices) == 0:
        # 如果没有故障清除后的数据，直接返回稳定
        return results
    t_check = t[t_check_indices]
    is_stable = True

    # --- 1. 功角稳定性分析 (Rotor Angle Stability) ---
    if ss.GENROU.n > 0:
        # 获取所有同步发电机的功角索引
        delta_indices = ss.GENROU.delta.a
        deltas_deg = ss.dae.ts.xy[np.ix_(t_check_indices, delta_indices)] * (180 / np.pi)

        spread_t = np.max(deltas_deg, axis=1) - np.min(deltas_deg, axis=1)

        violation_indices = np.where(spread_t > threshold_angle)[0]

        if len(violation_indices) > 0:
            is_stable = False
            results["Reason"].append("Angle Instability")
            # 取第一个越限时刻
            first_idx = violation_indices[0]
            t_event = t_check[first_idx]
            results["Time_Angle_Unstable"] = t_event
            results["Details"].append(f"Angle spread > {threshold_angle} at t={t_event:.3f}s")
    # --- 2. 频率稳定性分析 (Frequency Stability) ---
    # 使用发电机转速 omega 来估算频率 (f = omega * 60)
    if ss.GENROU.n > 0:
        omega_indices = ss.GENROU.omega.a
        freqs = ss.dae.ts.xy[np.ix_(t_check_indices, omega_indices)] *60.0

        f_low = 60 - threshold_freq_dev
        f_high = 60 + threshold_freq_dev

        any_low = np.any(freqs < f_low, axis=1)
        any_high = np.any(freqs > f_high, axis=1)

        violation_indices = np.where(any_low | any_high)[0]
        
        if len(violation_indices) > 0:
            is_stable = False
            results["Reason"].append("Frequency Instability")
            first_idx = violation_indices[0]
            t_event = t_check[first_idx]
            results["Time_Freq_Unstable"] = t_event
            results["Details"].append(f"Freq out of range at t={t_event:.3f}s")

    # --- 3. 电压稳定性分析 (Voltage Stability) ---
    # 检查所有母线电压
    threshold_volt_low = 1.0 - threshold_volt_dev
    threshold_volt_high = 1.0 + threshold_volt_dev
    if ss.Bus.n > 0:
        v_indices = np.array(ss.TDS.plotter.find('v Bus')[0])-1
        voltages = ss.dae.ts.xy[:, v_indices]
        
        end_window_mask = np.where(t > (t[-1] - 1.0))[0] # 取最后1秒的数据索引用于稳态电压检查
        
        # 3.2 长期电压恢复 (Long-term / Post-contingency)
        # 检查最后1秒的平均电压是否恢复到阈值以上
        if len(end_window_mask) > 0:
            final_voltages = np.mean(voltages[end_window_mask, :], axis=0)
            min_final_v = np.min(final_voltages)
            max_final_v = np.max(final_voltages)
            if min_final_v < threshold_volt_low or max_final_v > threshold_volt_high:
                is_stable = False
                results["Reason"].append("Voltage Instability")
                results["Details"].append(f"Post-fault voltage detected at {min_final_v:.3f} p.u.")
                results["Time_Volt_Unstable"] = t[-1] # 长期恢复失败，时间点不明确

    if not is_stable:
        results["Status"] = "Unstable"
    
    return results

def visualize_stats(stats_data, total_samples):
    """
    绘制简单的统计柱状图
    """
    df_stats = pd.DataFrame(stats_data)
    
    # 设置绘图风格
    plt.figure(figsize=(10, 6))
    
    # 创建柱状图
    bars = plt.bar(df_stats["Category"], df_stats["Count"], color=['green', 'orange', 'red', 'purple'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_samples) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({percentage:.1f}%)',
                 ha='center', va='bottom')

    plt.title(f'Simulation Stability Statistics (Total N={total_samples})')
    plt.ylabel('Number of Cases')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存统计图
    plt.savefig('Simulation_Stats_Chart.png', dpi=300)
    print("\n[INFO] 统计图表已保存为 'Simulation_Stats_Chart.png'")
    # plt.show() # 如果在非图形界面环境运行，请注释掉此行

def analyze_simulation_stats(csv_path="Total_Simulation_Summary.csv"):
    # 1. 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 未找到文件 {csv_path}。请先运行仿真生成汇总表。")
        return

    # 2. 读取 CSV
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    if total_samples == 0:
        print("警告: 汇总表为空。")
        return

    print(f"========================================")
    print(f"       仿真结果统计 (总样本数: {total_samples})")
    print(f"========================================")

    # 3. 定义要统计的列名 (对应你之前代码中的列名)
    categories = {
        "稳定 (Stable)": "Is_Stable",
        "电压失稳 (Voltage Unstable)": "Is_Volt_Unstable",
        "频率失稳 (Frequency Unstable)": "Is_Freq_Unstable",
        "功角失稳 (Angle Unstable)": "Is_Angle_Unstable"
    }

    stats_data = []

    # 4. 循环计算计数和比例
    for label, col_name in categories.items():
        if col_name in df.columns:
            count = df[col_name].sum() # Boolean求和: True=1, False=0
            percentage = (count / total_samples) * 100
            
            print(f"{label:<30} | 数量: {count:<5} | 占比: {percentage:.2f}%")
            
            stats_data.append({
                "Category": label.split(' (')[0], # 简化图表标签
                "Count": count,
                "Percentage": percentage
            })
        else:
            print(f"警告: 列 {col_name} 不存在于 CSV 中")

    print("========================================")
    print("注意: 失稳类型可能存在重叠 (例如一个样本可能同时发生电压和频率失稳)，\n因此失稳类型的百分比之和可能大于 100%。")

    # --- 5. 可视化 (画柱状图) ---
    visualize_stats(stats_data, total_samples)

def extract_line_flows_manually(ss,save_path):
    """
    根据 ANDES Line 模型文档手动计算支路潮流 (P, Q)。
    
    原理：
    S_ij = V_i * conj(I_ij)
    其中 I_ij 基于 Pi 型等值电路计算，考虑变压器变比 (tap) 和移相角 (phi)。
    """
    output_dir = os.path.join("line_flow",save_path)

    # 1. 准备电压数据
    # ------------------------------------------------------------------
    v_indices = ss.Bus.v.a
    a_indices = ss.Bus.a.a
    
    V_mag = ss.dae.ts.y[:, v_indices]
    Theta = ss.dae.ts.y[:, a_indices]
    
    # 构建复数电压矩阵 [Time, n_Buses]
    V_complex_all = V_mag * np.exp(1j * Theta)
    n_steps = V_complex_all.shape[0]
    
    # Bus ID -> Matrix Index 映射
    bus_id_map = {uid: i for i, uid in enumerate(ss.Bus.idx.v)}
    
    # 2. 准备线路参数
    # ------------------------------------------------------------------
    r = np.array(ss.Line.r.v)
    x = np.array(ss.Line.x.v)
    b = np.array(ss.Line.b.v)
    g = np.array(ss.Line.g.v)
    b1 = np.array(ss.Line.b1.v)
    g1 = np.array(ss.Line.g1.v)
    tap = np.array(ss.Line.tap.v)
    phi = np.array(ss.Line.phi.v)
    u_status = np.array(ss.Line.u.v)
    
    bus1_indices = np.array(ss.Line.bus1.v, dtype=int)
    bus2_indices = np.array(ss.Line.bus2.v, dtype=int)
    line_names = ss.Line.idx.v # 或者是 ss.Line.name.v
    
    n_lines = ss.Line.n
    
    # 初始化结果数组: [Time, Lines, 2]
    # flows[:, :, 0] 是 P, flows[:, :, 1] 是 Q
    flows = np.zeros((n_steps, n_lines, 2), dtype=np.float32)
    
    # 3. 批量计算所有线路潮流
    # ------------------------------------------------------------------
    # 为了加速，尽量向量化操作。如果不熟悉全矩阵操作，循环线路也可以
    
    for i in range(n_lines):
        # 如果线路断开，保持为 0
        if u_status[i] == 0:
            continue
            
        uid1 = bus1_indices[i]
        uid2 = bus2_indices[i]
        
        # 获取两端电压
        if uid1 in bus_id_map and uid2 in bus_id_map:
            idx1 = bus_id_map[uid1]
            idx2 = bus_id_map[uid2]
            
            V1 = V_complex_all[:, idx1]
            V2 = V_complex_all[:, idx2]
            
            # 参数计算
            y_series = 1.0 / (r[i] + 1j * x[i] + 1e-8)
            t_eff = tap[i] * np.exp(1j * phi[i])
            y_shunt_from = (g1[i] + 0.5 * g[i]) + 1j * (b1[i] + 0.5 * b[i])
            
            # 潮流计算
            V1_line = V1 / t_eff
            I_series = (V1_line - V2) * y_series
            I_shunt = V1_line * y_shunt_from
            I_line_in = I_series + I_shunt
            S_flow = V1_line * np.conj(I_line_in)
            
            # 填入数组
            flows[:, i, 0] = np.real(S_flow) # P
            flows[:, i, 1] = np.imag(S_flow) # Q

    # 4. 保存数据
    # ------------------------------------------------------------------
    # npy_path = os.path.join(output_dir, f"{file_prefix}.npy")
    np.save(output_dir, flows)
    
    # 5. 保存映射关系 (可选，只需在第一次运行时保存)
    # 检查是否已存在 mapping 文件，如果没有则保存
    mapping_path = os.path.join("line_flow", "lines_mapping.csv")
    if not os.path.exists(mapping_path):
        mapping_df = pd.DataFrame({
            "Line_Index": range(n_lines),
            "Line_Name": line_names,
            "From_Bus": bus1_indices,
            "To_Bus": bus2_indices
        })
        mapping_df.to_csv(mapping_path, index=False)
        print(f"[INFO] 线路映射表已保存: {mapping_path}")


    return output_dir

def run_voltage_stability_analysis():
    try:
        case_path = andes.utils.get_case(r'H:\andes-master\andes\cases\ieee39\ieee39_regcp1_motor_1.xlsx')
        print(f"使用算例: {case_path}")
    except IndexError:
        print("未找到内置的 IEEE 39 算例，请确保 ANDES 安装完整或指定具体文件路径。")
        return
    summary_list = []

    load_scales = [0.8,0.9,1.0,1.1,1.2] # 负载水平
    dura_times = [0.05,0.1,0.15,0.2,0.25,0.3] # 故障持续时间
    bus_nums = range(1,39) # 故障母线编号
    induction_motor_ratios = -1
    for scale in load_scales:
        # 按照负荷水平循环
        scale_label = f"{scale:.1f}"
        print(f"\n[INFO] 开始模拟: 负载水平 = {scale_label} x 基准值")
        for dura_time in dura_times:
            # 按照故障持续时间循环
            for bus_num in bus_nums:
                # 按照故障母线循环
                ss = andes.load(case_path, setup=False, default_config=True)


                ss.add('Fault',dict(bus=bus_num,tf=5.0,tc=5.0+dura_time,xf=0.0001,rf=0.0))
                print(ss.Fault.as_df())
            
                # 直接修改 PQ 模型的参数数组 p0 (有功) 和 q0 (无功)
                if ss.PQ.n > 0:
                    for i in range(ss.PQ.n):
                        original_p0 = ss.PQ.get("p0",f"PQ_{i+1}")
                        ss.PQ.alter("p0", f"PQ_{i+1}", original_p0 * scale)
                        
                        original_q0 = ss.PQ.get("q0",f"PQ_{i+1}")
                        ss.PQ.alter("q0", f"PQ_{i+1}", original_q0 * scale)
                    print(f"  - 已将所有 PQ 负载调整为 {scale_label} 倍")
                else:
                    print("  - 警告: 系统中没有找到 PQ 负载")

                if ss.PV.n > 0:
                    for i in range(ss.PV.n):
                        original_p0 = ss.PV.get("p0",i+1)
                        ss.PV.alter("p0", i+1, original_p0 * scale)
                        
                        original_q0 = ss.PV.get("q0",i+1)
                        ss.PV.alter("q0", i+1, original_q0 * scale)
                    print(f"  - 已将所有 PV 发电机有功调整为 {scale_label} 倍")
                else:
                    print("  - 警告: 系统中没有找到 PV 发电机")

                ss.setup()
                ss.PFlow.run()
                
                ss.TDS.config.tf = 20      # simulate for 5 seconds to save time
                ss.TDS.config.tstep = 0.01   # time step
                ss.TDS.config.no_tqdm = 1  # disable progres bar printing 
                ss.TDS.config.criteria = 0
                ss.TDS.run()

                # df_full = ss.dae.ts.df
                # _ = extract_line_flows_manually(ss,f"Line_Flows_load_{scale_label}_Bus_{bus_num}_Dur_{dura_time}.npy")
                # df_full = pd.concat([df_ts, df_flows], axis=1)
                # df_full = df_full.round(4) # 保留四位小数
                ss.TDS.load_plotter()
                # fig, ax = ss.TDS.plt.plot(ss.TDS.plotter.find('delta GENROU')[0])
                # fig.savefig(f"figures/ieee39_Fault_Bus_{bus_num}_load_{scale_label}_delta.png")
                results = analyze_stability(ss,dura_time)
                print(results)
                reasons = results.get("Reason",[])
                
                df_full = ss.dae.ts.df
                df_full = df_full.round(4) # 保留四位小数
                v_indices = np.array(ss.TDS.plotter.find('v Bus')[0])-1
                a_indices = np.array(ss.TDS.plotter.find('a Bus')[0])-1
                delta_GENROU_indices = np.array(ss.TDS.plotter.find('delta GENROU')[0])-1
                omega_GENROU_indices = np.array(ss.TDS.plotter.find('omega GENROU')[0])-1
                indices = np.hstack([v_indices,a_indices,delta_GENROU_indices,omega_GENROU_indices])
                df_full = df_full.iloc[:,indices]

                df_full['Label_Stable'] = (results["Status"] == "Stable")
                df_full['Label_Angle'] = ("Angle Instability" in reasons)
                df_full['Label_Freq'] = ("Frequency Instability" in reasons)
                df_full['Label_Volt'] = ("Voltage Instability" in reasons)
                df_full['Time_Angle_Unstable'] = results['Time_Angle_Unstable'] if results['Time_Angle_Unstable'] else -1
                df_full['Time_Freq_Unstable'] = results['Time_Freq_Unstable'] if results['Time_Freq_Unstable'] else -1
                df_full['Time_Volt_Unstable'] = results['Time_Volt_Unstable'] if results['Time_Volt_Unstable'] else -1

                
                # 5. 添加元数据列 (故障时间、参数、标签)
                df_full['Fault_Duration'] = dura_time
                df_full['Fault_Bus'] = bus_num
                df_full['Load_Scale'] = scale
                df_full['Induction_Motor_Ratio'] = induction_motor_ratios
                
                csv_name = f"IEEE39_regcp1_motor_1/ieee39_ThreePhase_Fault_load_{scale_label}_Bus_{bus_num}_Dur_{dura_time}_Induction_{induction_motor_ratios}_results.csv"
                df_full.to_csv(csv_name, index=False)
                
                summary_list.append({
                    "Bus": bus_num,
                    "Duration": dura_time,
                    "Scale": scale,
                    "Is_Stable": (results["Status"] == "Stable"),
                    "Is_Volt_Unstable": ("Voltage Instability" in reasons),
                    "Is_Freq_Unstable": ("Frequency Instability" in reasons),
                    "Is_Angle_Unstable": ("Angle Instability" in reasons),
                    "Details": str(results["Details"])
                })
                # 循环结束后，保存总表
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv("Total_Simulation_Summary.csv", index=False)



if __name__ == "__main__":
    run_voltage_stability_analysis()
    analyze_simulation_stats()
