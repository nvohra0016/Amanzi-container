#include <UnitTest++.h>

#include <stdexcept>
#include <memory>
#include <errno.h>

#include "../Exodus_file.hh"
#include "../Exodus_readers.hh"

#include "Parameters.hh"
#include "Side_set.hh"

extern std::string test_file_path(const std::string& fname);
extern std::string split_file_path(const std::string& fname);

struct Exodus_file_holder
{
    ExodusII::Exodus_file file;
    std::auto_ptr<Mesh_data::Data> data;

    Exodus_file_holder (const char* filename) :
        file (filename),
        data (ExodusII::read_exodus_file (filename))
    {  }
};

struct Big_File : Exodus_file_holder
{
  Big_File () : Exodus_file_holder (test_file_path("htc_rad_test-random.exo").c_str()) { }
};

struct quad_4x4 : Exodus_file_holder
{
    quad_4x4 () : Exodus_file_holder (test_file_path("quad_4x4_ss.exo").c_str()) { }
};

struct hex_split_2_0 : Exodus_file_holder
{
     hex_split_2_0() : Exodus_file_holder (split_file_path("hex_4x4x4_ss.par.2.0").c_str()) { }
};

struct hex_split_2_1 : Exodus_file_holder
{
     hex_split_2_1() : Exodus_file_holder (split_file_path("hex_4x4x4_ss.par.2.1").c_str()) { }
};


struct hex11_split_2_0 : Exodus_file_holder
{
     hex11_split_2_0() : Exodus_file_holder (split_file_path("hex_11x11x11_ss.par.2.0").c_str()) { }
};

struct hex11_split_2_1 : Exodus_file_holder
{
     hex11_split_2_1() : Exodus_file_holder (split_file_path("hex_11x11x11_ss.par.2.1").c_str()) { }
};


struct twoblktet_2_0 : Exodus_file_holder
{
     twoblktet_2_0() : Exodus_file_holder (split_file_path("twoblktet_ss.par.2.0").c_str()) { }
};


struct twoblktet_2_1 : Exodus_file_holder
{
     twoblktet_2_1() : Exodus_file_holder (split_file_path("twoblktet_ss.par.2.1").c_str()) { }
};



SUITE (Big_File)
{

    TEST_FIXTURE (Big_File, Parameters)
    {
        const Mesh_data::Parameters &params (data->parameters ());

        CHECK_EQUAL (params.element_block_ids_.size (), 3);
        CHECK_EQUAL (params.node_set_ids_.size (), 0);
        CHECK_EQUAL (params.side_set_ids_.size (), 4);

        CHECK_EQUAL (params.dimensions_, 3);
        CHECK_EQUAL (params.num_nodes_, 6615);
        CHECK_EQUAL (params.num_elements_, 5600);
        CHECK_EQUAL (params.num_element_blocks_, 3);
        CHECK_EQUAL (params.num_node_sets_, 0);
        CHECK_EQUAL (params.num_side_sets_, 4);
    };

    TEST_FIXTURE (Big_File, Side_Sets)
    {

        {
            const Mesh_data::Side_set &side (data->side_set (0));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 400);
            // CHECK_EQUAL (side.num_nodes (), 1600);
        }

        {
            const Mesh_data::Side_set &side (data->side_set (1));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 400);
            // CHECK_EQUAL (side.num_nodes (), 1600);
        }

        {
            const Mesh_data::Side_set &side (data->side_set (2));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 1120);
            // CHECK_EQUAL (side.num_nodes (), 4480);
        }

        {
            const Mesh_data::Side_set &side (data->side_set (3));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 800);
            // CHECK_EQUAL (side.num_nodes (), 3200);
        }

    }

}

SUITE (quad_4x4)
{

    TEST_FIXTURE (quad_4x4, Parameters)
    {
        const Mesh_data::Parameters &params (data->parameters ());

        CHECK_EQUAL (params.element_block_ids_.size (), 2);
        CHECK_EQUAL (params.node_set_ids_.size (), 11);
        CHECK_EQUAL (params.side_set_ids_.size (), 11);

        CHECK_EQUAL (params.dimensions_, 3);
        CHECK_EQUAL (params.num_nodes_, 16);
        CHECK_EQUAL (params.num_elements_, 9);
        CHECK_EQUAL (params.num_element_blocks_, 2);
        CHECK_EQUAL (params.num_node_sets_, 11);
        CHECK_EQUAL (params.num_side_sets_, 11);
    };

    TEST_FIXTURE (quad_4x4, Side_Sets)
    {

        {
            const Mesh_data::Side_set &side (data->side_set (0));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 12);
            // CHECK_EQUAL (side.num_nodes (), 1600);
        }

        {
            const Mesh_data::Side_set &side (data->side_set (1));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 3);
            // CHECK_EQUAL (side.num_nodes (), 1600);
        }

        {
            const Mesh_data::Side_set &side (data->side_set (2));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 3);
            // CHECK_EQUAL (side.num_nodes (), 4480);
        }

        {
            const Mesh_data::Side_set &side (data->side_set (3));

            // CHECK       (side.has_node_factors ());
            CHECK_EQUAL (side.num_sides (), 3);
            // CHECK_EQUAL (side.num_nodes (), 3200);
        }

    }



}

SUITE (hex_split)
{
  TEST_FIXTURE (hex_split_2_0, Parameters)
  {
    const Mesh_data::Parameters &params (data->parameters ());

    CHECK_EQUAL (params.element_block_ids_.size (), 3);
    CHECK_EQUAL (params.node_set_ids_.size (), 21);
    CHECK_EQUAL (params.side_set_ids_.size (), 21);

    CHECK_EQUAL (params.dimensions_, 3);
    CHECK_EQUAL (params.num_nodes_, 42);
    CHECK_EQUAL (params.num_elements_, 13);
    CHECK_EQUAL (params.num_element_blocks_, 3);
    CHECK_EQUAL (params.num_node_sets_, 21);
    CHECK_EQUAL (params.num_side_sets_, 21);

  }
  TEST_FIXTURE (hex_split_2_1, Parameters)
  {
    const Mesh_data::Parameters &params (data->parameters ());

    CHECK_EQUAL (params.element_block_ids_.size (), 3);
    CHECK_EQUAL (params.node_set_ids_.size (), 21);
    CHECK_EQUAL (params.side_set_ids_.size (), 21);

    CHECK_EQUAL (params.dimensions_, 3);
    CHECK_EQUAL (params.num_nodes_, 43);
    CHECK_EQUAL (params.num_elements_, 14);
    CHECK_EQUAL (params.num_element_blocks_, 3);
    CHECK_EQUAL (params.num_node_sets_, 21);
    CHECK_EQUAL (params.num_side_sets_, 21);

  }
}

SUITE (hex11_split)
{
  TEST_FIXTURE (hex11_split_2_0, Parameters)
  {
    const Mesh_data::Parameters &params (data->parameters ());

    CHECK_EQUAL (params.element_block_ids_.size (), 3);
    CHECK_EQUAL (params.node_set_ids_.size (), 20);
    CHECK_EQUAL (params.side_set_ids_.size (), 20);

    CHECK_EQUAL (params.dimensions_, 3);
    CHECK_EQUAL (params.num_nodes_, 726);
    CHECK_EQUAL (params.num_elements_, 500);
    CHECK_EQUAL (params.num_element_blocks_, 3);
    CHECK_EQUAL (params.num_node_sets_, 20);
    CHECK_EQUAL (params.num_side_sets_, 20);

  }
  TEST_FIXTURE (hex11_split_2_1, Parameters)
  {
    const Mesh_data::Parameters &params (data->parameters ());

    CHECK_EQUAL (params.element_block_ids_.size (), 3);
    CHECK_EQUAL (params.node_set_ids_.size (), 20);
    CHECK_EQUAL (params.side_set_ids_.size (), 20);

    CHECK_EQUAL (params.dimensions_, 3);
    CHECK_EQUAL (params.num_nodes_, 726);
    CHECK_EQUAL (params.num_elements_, 500);
    CHECK_EQUAL (params.num_element_blocks_, 3);
    CHECK_EQUAL (params.num_node_sets_, 20);
    CHECK_EQUAL (params.num_side_sets_, 20);

  }
}

SUITE (twoblktet)
{
  TEST_FIXTURE (twoblktet_2_0, Parameters)
  {
    const Mesh_data::Parameters &params (data->parameters ());

    CHECK_EQUAL (params.element_block_ids_.size (), 2);
    CHECK_EQUAL (params.node_set_ids_.size (), 17);
    CHECK_EQUAL (params.side_set_ids_.size (), 17);

    CHECK_EQUAL (params.dimensions_, 3);
    CHECK_EQUAL (params.num_nodes_, 139);
    CHECK_EQUAL (params.num_elements_, 407);
    CHECK_EQUAL (params.num_element_blocks_, 2);
    CHECK_EQUAL (params.num_node_sets_, 17);
    CHECK_EQUAL (params.num_side_sets_, 17);

  }
  TEST_FIXTURE (twoblktet_2_1, Parameters)
  {
    const Mesh_data::Parameters &params (data->parameters ());

    CHECK_EQUAL (params.element_block_ids_.size (), 2);
    CHECK_EQUAL (params.node_set_ids_.size (), 17);
    CHECK_EQUAL (params.side_set_ids_.size (), 17);

    CHECK_EQUAL (params.dimensions_, 3);
    CHECK_EQUAL (params.num_nodes_, 143);
    CHECK_EQUAL (params.num_elements_, 406);
    CHECK_EQUAL (params.num_element_blocks_, 2);
    CHECK_EQUAL (params.num_node_sets_, 17);
    CHECK_EQUAL (params.num_side_sets_, 17);

  }
}



