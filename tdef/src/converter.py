import pandas as pd
import csv
import argparse

k_name = "Kernel Name"
m_name = "Metric Name"
m_value = "Metric Value"
m_names = ["lts__t_sectors_op_write.sum", "lts__t_sectors_op_read.sum"]
replacement_m_names = ["l2_write_transactions","l2_read_transactions"]
boilerplate = """==4037416== NVPROF is profiling process 4037416, command: python pinch/src/attacks/DeepSniffer/deepsniffer/ProfileModels/profile_model.py --source torchvision --name resnet18 --dataset imagenet --checkpoint resnet18 --framework pytorch --compiler tvm\n==4037416== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.\n==4037416== Profiling application: python pinch/src/attacks/DeepSniffer/deepsniffer/ProfileModels/profile_model.py --source torchvision --name resnet18 --dataset imagenet --checkpoint resnet18 --framework pytorch --compiler tvm\n==4037416== Profiling result:\n"""
    
def parse_args():
    parser = argparse.ArgumentParser(description='Read input file and write to output file')

    parser.add_argument('input_file', help='Path to the input file')
    parser.add_argument('output_file', help='Path to the output file')

    args = parser.parse_args()

    return args

def convert(input_file_path, output_file_path):
    with pd.option_context("mode.chained_assignment", None):
        df = pd.read_csv(input_file_path, skiprows=6, thousands=",")
        df = df[[k_name, m_name, m_value]].copy()

        m_0 = df[df[m_name] == m_names[0]]
        m_1 = df[df[m_name] == m_names[1]]

        m_1[m_value + "_"] = m_0[m_value].to_numpy()

        m_1 = m_1.rename(columns={k_name: "Kernel", m_value: replacement_m_names[0], m_value+"_": replacement_m_names[1]})
        del(m_1[m_name])

        emulate_labels_vals = {
            "Device": "Tesla V100-PCIE-32GB (0)",
            "Context": "1",
            "Stream": "7",
            "Correlation_ID": 848
        }

        final_df_rows = []
        for index, row in m_1.iterrows():
            final_df_rows.append({
                "Device": emulate_labels_vals["Device"],
                "Context": emulate_labels_vals["Context"],
                "Stream": emulate_labels_vals["Stream"],
                "Kernel": row["Kernel"],
                "Correlation_ID": emulate_labels_vals["Correlation_ID"] + index,
                replacement_m_names[0]: row[replacement_m_names[0]],
                replacement_m_names[1]: row[replacement_m_names[1]]
            })

        final_df = pd.DataFrame(final_df_rows)
        final_df.to_csv(output_file_path, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        with open(output_file_path, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            f.write(boilerplate)
            for line in lines:
                f.write(line)

if __name__ == "__main__":
    args = parse_args()
    convert(args.input_file, args.output_file)