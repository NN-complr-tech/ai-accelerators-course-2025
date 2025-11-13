import subprocess
import matplotlib.pyplot as plt

name_of_proc = "C:\\dev\\Source\\ai-accelerators-course-2025\\tasks\\02-softmax-cuda\\chuvashev\\output.exe"
name_of_bat = "C:\\dev\\Source\\ai-accelerators-course-2025\\tasks\\02-softmax-cuda\\chuvashev\\script.bat"
values = ["1024", "2048", "4069", "12000", "15000", "20000"]
threads = ["64", "128", "256", "512", "1024"]
colors = ['red', 'blue', 'green', 'orange', "purple"]
plt.figure(figsize=(10, 6))

temp = 0

for thread in threads:
    cmd_compile= [name_of_bat, thread]
    result_compile = subprocess.run(cmd_compile, capture_output=True, text=True)
    simt_times = []
    print(f"Return code: {result_compile.returncode}")
    print(f"Output: {result_compile.stdout}")
    if result_compile.stderr:
        print(f"Errors: {result_compile.stderr}")
    for value in values:
        cmd = [name_of_proc, value]
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        current_result = []
        lines = result.stdout.split('\n')
        simt_times.append(float(lines[0]) / (float(value) * float(value)))
    sizes = [int(v) for v in values]
    plt.plot(sizes, simt_times, marker='x', label='SIMT', color=colors[temp])
    temp+=1

plt.xlabel("Количество элементов (n)")
plt.ylabel("Время (сек)")
plt.title("Сравнение методов вычисления Softmax")
plt.legend()
plt.grid(True)

plt.savefig("softmax_comparison.png", dpi=300)