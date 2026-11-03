# FilesystemBackend 跨进程持久化验证

日期：2026-10-13。vm-60课程主应用环境：Python3.13.13、Deep Agents0.7.8、DeepSeek-V4-Flash。

配置：FilesystemBackend(root_dir=/home/simin/course09-quanty/persistent-reports/task1-persist-02, virtual_mode=True)。工具路径/report.md映射到该目录的report.md；这是文件路径映射，不代表完整操作系统沙箱。

1. 写入进程PID1506526：analyze_price_volume(window=3) → write_file(report.md) → write_todos →正常退出。模型仍调用了规划工具，虽然提示称简单查询无需规划，可作为后续精简Harness的观察点。
2. 新进程PID1506525：仅传入1条新消息，要求read_file读取/report.md。未传旧消息或报告内容，只调用read_file并成功返回。
3. 两进程检查的磁盘文件存在，SHA256相同：03b4661f46bb1f835156c05fa1f7058e89be0898257f6b8af65c5c02ffadd5ba。
4. 离线verify_persistence.py核验调用ID对应、读回各非空行与磁盘内容相符。

证据：persistence-write.json、persistence-read.json、persistence-verification.json。persistent-report.md是远端磁盘报告的本地副本。

首次task1-persist-01尝试遇到网关502，未计为成功；成功实验使用新的task1-persist-02目录。

结论：验证了同一VM上的Python进程退出后，新进程可以通过FilesystemBackend读取原磁盘文件。没有重启VM，也没有验证磁盘故障恢复、备份或跨机器迁移。
