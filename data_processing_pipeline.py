from BrainMRIDataLoader import BrainMRIDataLoader

load_data = BrainMRIDataLoader(base_path='images', categories=['Healthy', 'Tumor'])
load_data.load_data()
load_data.show_summary()