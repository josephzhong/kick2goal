import tensorflow as tf

def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame."""

    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    data = []

    for event in summary_iterator(root_dir):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                data.append({
                    'wall_time': event.wall_time,
                    'name': value.tag,
                    'step': event.step,
                    'value': value.simple_value
                })

    df = pd.DataFrame(data)

    if sort_by is not None:
        df = df.sort_values(sort_by)

    return df

# Example usage:
df = convert_tb_data('tb_log/PPO_150/events.out.tfevents.1730751159.JosephdeMacBook-Air.local.20368.0')
print(df)