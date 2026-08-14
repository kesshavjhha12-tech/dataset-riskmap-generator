"""
Configuration for GCM-HAIRNet Dataset Generation
"""

AOIS = {
    # ============================================================
    # SOUTH INDIA (15)
    # ============================================================
    'Bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'category': 'metropolitan'},
    'Mysuru': {'lat': 12.2958, 'lon': 76.6394, 'category': 'metropolitan'},
    'Mangaluru': {'lat': 12.9141, 'lon': 74.8560, 'category': 'metropolitan'},
    'Hubballi': {'lat': 15.3647, 'lon': 75.1240, 'category': 'metropolitan'},
    'Belagavi': {'lat': 15.8497, 'lon': 74.4977, 'category': 'metropolitan'},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'category': 'metropolitan'},
    'Warangal': {'lat': 17.9686, 'lon': 79.5941, 'category': 'metropolitan'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'category': 'metropolitan'},
    'Coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'category': 'metropolitan'},
    'Madurai': {'lat': 9.9252, 'lon': 78.1198, 'category': 'metropolitan'},
    'Kochi': {'lat': 9.9312, 'lon': 76.2673, 'category': 'metropolitan'},
    'Thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'category': 'metropolitan'},
    'Kozhikode': {'lat': 11.2588, 'lon': 75.7804, 'category': 'metropolitan'},
    'Vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'category': 'metropolitan'},
    'Visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'category': 'metropolitan'},

    # ============================================================
    # WEST INDIA (10)
    # ============================================================
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'category': 'metropolitan'},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'category': 'metropolitan'},
    'Nagpur': {'lat': 21.1458, 'lon': 79.0882, 'category': 'metropolitan'},
    'Nashik': {'lat': 19.9975, 'lon': 73.7898, 'category': 'metropolitan'},
    'Aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'category': 'metropolitan'},
    'Surat': {'lat': 21.1702, 'lon': 72.8311, 'category': 'metropolitan'},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'category': 'metropolitan'},
    'Vadodara': {'lat': 22.3072, 'lon': 73.1812, 'category': 'metropolitan'},
    'Rajkot': {'lat': 22.3039, 'lon': 70.8022, 'category': 'metropolitan'},
    'Panaji': {'lat': 15.4909, 'lon': 73.8278, 'category': 'metropolitan'},

    # ============================================================
    # NORTH INDIA (13)
    # ============================================================
    'Delhi': {'lat': 28.7041, 'lon': 77.1025, 'category': 'metropolitan'},
    'Gurugram': {'lat': 28.4595, 'lon': 77.0266, 'category': 'metropolitan'},
    'Noida': {'lat': 28.5355, 'lon': 77.3910, 'category': 'metropolitan'},
    'Ghaziabad': {'lat': 28.6692, 'lon': 77.4538, 'category': 'metropolitan'},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'category': 'metropolitan'},
    'Chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'category': 'metropolitan'},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462, 'category': 'metropolitan'},
    'Kanpur': {'lat': 26.4499, 'lon': 80.3319, 'category': 'metropolitan'},
    'Varanasi': {'lat': 25.3176, 'lon': 82.9739, 'category': 'metropolitan'},
    'Agra': {'lat': 27.1767, 'lon': 78.0081, 'category': 'metropolitan'},
    'Dehradun': {'lat': 30.3165, 'lon': 78.0322, 'category': 'semi_urban'},
    'Amritsar': {'lat': 31.6340, 'lon': 74.8723, 'category': 'metropolitan'},
    'Ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'category': 'metropolitan'},

    # ============================================================
    # EAST INDIA (8)
    # ============================================================
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'category': 'metropolitan'},
    'Bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'category': 'semi_urban'},
    'Cuttack': {'lat': 20.4625, 'lon': 85.8830, 'category': 'semi_urban'},
    'Guwahati': {'lat': 26.1445, 'lon': 91.7362, 'category': 'metropolitan'},
    'Patna': {'lat': 25.5941, 'lon': 85.1376, 'category': 'metropolitan'},
    'Ranchi': {'lat': 23.3441, 'lon': 85.3096, 'category': 'metropolitan'},
    'Jamshedpur': {'lat': 22.8046, 'lon': 86.2029, 'category': 'industrial'},
    'Siliguri': {'lat': 26.7271, 'lon': 88.3953, 'category': 'semi_urban'},

    # ============================================================
    # CENTRAL INDIA (5)
    # ============================================================
    'Bhopal': {'lat': 23.2599, 'lon': 77.4126, 'category': 'metropolitan'},
    'Indore': {'lat': 22.7196, 'lon': 75.8577, 'category': 'metropolitan'},
    'Raipur': {'lat': 21.2514, 'lon': 81.6296, 'category': 'metropolitan'},
    'Gwalior': {'lat': 26.2183, 'lon': 78.1828, 'category': 'metropolitan'},

    # ============================================================
    # ADDITIONAL DIVERSE URBAN SCENES (9)
    # ============================================================
    'Srinagar': {'lat': 34.0837, 'lon': 74.7973, 'category': 'semi_urban'},
    'Jammu': {'lat': 32.7266, 'lon': 74.8570, 'category': 'semi_urban'},
    'Shimla': {'lat': 31.1048, 'lon': 77.1734, 'category': 'rural'},
    'Thiruvalla': {'lat': 9.3856, 'lon': 76.5745, 'category': 'semi_urban'},
    'Tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'category': 'metropolitan'},
    'Salem': {'lat': 11.6643, 'lon': 78.1460, 'category': 'metropolitan'},
    'Rajahmundry': {'lat': 17.0005, 'lon': 81.8040, 'category': 'metropolitan'},
    'Meerut': {'lat': 28.9845, 'lon': 77.7064, 'category': 'metropolitan'},
    'Kota': {'lat': 25.2138, 'lon': 75.8648, 'category': 'metropolitan'},
}

# ============================================================
# Configuration Parameters
# ============================================================

AOI_SIZE_KM = 32
CELL_SIZE_KM = 1
GRID_SIZE = 32

GIS_FEATURES = [
    'population_density', 'building_density', 'road_density',
    'road_intersection_density', 'distance_to_highway',
    'builtup_percentage', 'vegetation_percentage', 'water_percentage',
    'night_lights', 'school_count', 'hospital_count',
    'police_count', 'bus_stop_count', 'elevation', 'slope',
    'ndvi', 'ndbi', 'commercial_percentage'
]

COMPONENT_INDICES = {
    'HE': [0, 1, 9, 10, 11, 12],
    'IC': [2, 3, 4, 13, 14],
    'UA': [8, 5, 17],
    'EB': [6, 7, 15, 16],
}

SMOOTH_SIGMA = 1.5
CNN_EPOCHS = 100
CNN_LR = 1e-3
DATA_ROOT = './data'