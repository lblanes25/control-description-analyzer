
import yaml

class ConfigManager:
    """Class for managing configuration settings"""

    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = {}

        if config_file:
            self.load_config(config_file)

    def load_config(self, path):
        """Load configuration from YAML file"""
        try:
            with open(path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")

    def get_element_config(self, name):
        """Get configuration for a specific element"""
        return self.config.get("elements", {}).get(name.upper(), {})

    def get_vague_terms(self):
        """Get list of vague terms from config"""
        return self.config.get("vague_terms", [])

    def get_column_defaults(self):
        """Get column mapping from config"""
        return self.config.get("columns", {})

    def get_type_keywords(self):
        """Get control type keywords from config"""
        return self.config.get("control_type_keywords", {})

    def get_audit_leader_column(self):
        """Get audit leader column name from config"""
        columns = self.get_column_defaults()
        # First look in columns section
        if columns and "audit_leader" in columns:
            return columns["audit_leader"]
        # Then check for top level config
        return self.config.get("audit_leader_column")